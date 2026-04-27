import numpy as np


def effective_counts(counts, cfg):
    return (counts["npairs"] * counts["weightavg"]).reshape(
        cfg.ns, cfg.nmu_bins
    )

def normalize_counts(counts, w):
    S = np.sum(w)
    S2 = np.sum(w**2)
    norm = S*S - S2
    if norm <= 0:
        raise ValueError("Normalization is non-positive")
        
    return counts / norm, norm

def landy_szalay_region(DD, DR, RR, wD, wR, cfg):

    DDm = effective_counts(DD, cfg)
    DRm = effective_counts(DR, cfg)
    RRm = effective_counts(RR, cfg)

    DDn, _ = normalize_counts(DDm, wD)
    RRn, rr_weight = normalize_counts(RRm, wR)

    SD = np.sum(wD)
    SR = np.sum(wR)

    DRn = DRm / (SD * SR)

    xi_smu = np.zeros_like(RRn)
    mask = RRn > 0
    xi_smu[mask] = (DDn[mask] - 2.0 * DRn[mask] + RRn[mask]) / RRn[mask]

    return xi_smu, rr_weight

def combine_xi(xi_dict, rr_weight_dict, acfg):

    regions = acfg.get_regions()

    num = np.zeros_like(xi_dict[regions[0]])
    denom = 0.0

    for r in regions:
        num += rr_weight_dict[r] * xi_dict[r]
        denom += rr_weight_dict[r]

    if denom == 0:
        raise ValueError("RR weight sum is zero")
    
    return num / denom

def monopole(xi_smu, cfg):
    xi_2d = xi_smu.reshape(cfg.ns, cfg.nmu_bins)
    return xi_2d.mean(axis=1)

def compute_xi(DD, DR, RR, wD, wR, cfg, acfg):

    xi_smu = {}
    rr_weight = {}

    regions = acfg.get_regions()

    for r in regions:
        xi_smu[r], rr_weight[r] = landy_szalay_region(
            DD[r], DR[r], RR[r],
            wD[r], wR[r],
            cfg
        )

    if acfg.combine_regions:
        xi = combine_xi(xi_smu, rr_weight, acfg)
        xi0 = monopole(xi, cfg)
    else:
        xi = xi_smu
        xi0 = {
            r: monopole(xi_smu[r], cfg)
            for r in regions
        }

    return xi, xi0

def combine_err(err_dict, rr_weight_dict, acfg):
    regions = acfg.get_regions()

    var = 0.0
    denom = 0.0

    for r in regions:
        w = rr_weight_dict[r]
        var += (w ** 2) * (err_dict[r] ** 2)
        denom += w

    var /= (denom ** 2)
    return np.sqrt(var)