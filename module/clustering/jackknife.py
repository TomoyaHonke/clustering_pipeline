import numpy as np
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler
from .counts import radecz_to_xyz, z_to_chi


def make_angular_jackknife_labels(
    data_ra,
    data_dec,
    rand_ra,
    rand_dec,
    acfg,
):
    subsampler = KMeansSubsampler(
        mode="angular",
        positions=(data_ra, data_dec),
        nsamples=acfg.n_jack,
        position_type="rd",
    )

    labels_data = subsampler.label((data_ra, data_dec), position_type="rd")
    labels_rand = subsampler.label((rand_ra, rand_dec), position_type="rd")

    return labels_data, labels_rand, subsampler


def make_pycorr_positions(ra, dec, z, cosmo, cfg):
    if cfg.mode == "xyz":
        x, y, zpos = radecz_to_xyz(ra, dec, z, cosmo)
        return (x, y, zpos), "xyz"

    elif cfg.mode == "radec":
        chi = z_to_chi(z, cosmo)
        return (ra, dec, chi), "rdd"

    else:
        raise ValueError("cfg.mode must be 'xyz' or 'radec'")

        
def compute_pycorr_jackknife_xi0(
    data_positions,
    random_positions,
    data_weights,
    random_weights,
    labels_data,
    labels_random,
    s_edges,
    mu_edges,
    position_type,
    cfg,
):
    res = TwoPointCorrelationFunction(
        "smu",
        edges=(s_edges, mu_edges),
        data_positions1=data_positions,
        randoms_positions1=random_positions,
        position_type=position_type,
        data_weights1=data_weights,
        randoms_weights1=random_weights,
        estimator="landyszalay",
        engine="corrfunc",
        nthreads=cfg.nthreads,
        data_samples1=labels_data,
        randoms_samples1=labels_random,
    )

    xi0 = np.asarray(res(ell=0)).squeeze()

    xi0_jk = np.array([
        np.asarray(res.realization(ii)(ell=0)).squeeze()
        for ii in res.realizations
    ])

    n_jack = xi0_jk.shape[0]
    cov_xi0 = (n_jack - 1) * np.cov(xi0_jk, rowvar=False, ddof=0)
    err_xi0 = np.sqrt(np.diag(cov_xi0))

    return xi0, xi0_jk, cov_xi0, err_xi0, res