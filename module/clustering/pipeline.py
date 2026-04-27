import numpy as np

from .io import load_data_catalog, load_random_catalog, select_zbin
from .counts import run_dd_dr_rr
from .estimator import compute_xi, combine_err
from .jackknife import make_angular_jackknife_labels, make_pycorr_positions, compute_pycorr_jackknife_xi0

def compute_monopole(
    data_path,
    random_path,
    cosmo,
    cfg,
    acfg,
):

    results = {}

    if acfg.zbins is None:
        zbin_items = {"all": (None, None)}.items()
    else:
        zbin_items = acfg.zbins.items()

    for zlabel, (zmin, zmax) in zbin_items:

        DD, DR, RR = {}, {}, {}
        wD, wR = {}, {}

        data_cache = {}
        rand_cache = {}
        
        for r in acfg.get_regions():
        
            dra, ddec, dz, dw = load_data_catalog(data_path[r])
            rra, rdec, rz, rw = load_random_catalog(random_path[r])
        
            dra, ddec, dz, dw = select_zbin(dra, ddec, dz, dw, zmin, zmax)
            rra, rdec, rz, rw = select_zbin(rra, rdec, rz, rw, zmin, zmax)
        
            data_cache[r] = (dra, ddec, dz, dw)
            rand_cache[r] = (rra, rdec, rz, rw)

            DD[r], DR[r], RR[r] = run_dd_dr_rr(
                dra, ddec, dz, dw,
                rra, rdec, rz, rw,
                cosmo,
                cfg,
            )

            wD[r] = dw
            wR[r] = rw

        xi_smu, xi0 = compute_xi(
            DD, DR, RR,
            wD, wR,
            cfg,
            acfg,
        )

        s = 0.5 * (cfg.binfile[:-1] + cfg.binfile[1:])

        if acfg.use_jackknife:
        
            err = {}
            rr_weight = {}
        
            for r in acfg.get_regions():

                dra, ddec, dz, dw = data_cache[r]
                rra, rdec, rz, rw = rand_cache[r]
                
                data_positions, position_type = make_pycorr_positions(dra, ddec, dz, cosmo, cfg)
                random_positions, _ = make_pycorr_positions(rra, rdec, rz, cosmo, cfg)
        
                labels_data, labels_rand, _ = make_angular_jackknife_labels(
                    dra, ddec,
                    rra, rdec,
                    acfg,
                )
        
                mu_edges = np.linspace(-cfg.mu_max, cfg.mu_max, cfg.nmu_bins + 1)
        
                xi0_pycorr, xi0_jk, cov, err_r, _ = compute_pycorr_jackknife_xi0(
                    data_positions=data_positions,
                    random_positions=random_positions,
                    data_weights=dw,
                    random_weights=rw,
                    labels_data=labels_data,
                    labels_random=labels_rand,
                    s_edges=cfg.binfile,
                    mu_edges=mu_edges,
                    position_type=position_type,
                    cfg=cfg,
                )
        
                err[r] = err_r

                SR = np.sum(rw)
                rr_weight[r] = SR * SR - np.sum(rw ** 2)
    
            if acfg.combine_regions:
                err = combine_err(err, rr_weight, acfg)
                
            results[zlabel] = (s, xi0, err)
            
        else:
            results[zlabel] = (s, xi0)

    return results