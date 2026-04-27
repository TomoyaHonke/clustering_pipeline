import numpy as np
import astropy.units as u
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
import Corrfunc

def z_to_chi(z, cosmo):
    return cosmo.comoving_distance(z).to(u.Mpc).value * cosmo.h

def radecz_to_xyz(ra, dec, z, cosmo):
    chi = z_to_chi(z, cosmo)

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    zpos = chi * np.sin(dec_rad)

    return x, y, zpos #Mpc/h

def run_dd_auto_xyz(x, y, z, weight, cfg):
    return Corrfunc.mocks.DDsmu_mocks(
        autocorr=1,
        nthreads=cfg.nthreads,
        binfile=cfg.binfile,
        mumax=cfg.mu_max,
        nmubins=cfg.nmu_bins,
        X1=x.astype(np.float64),
        Y1=y.astype(np.float64),
        Z1=z.astype(np.float64),
        weights1=weight.astype(np.float64),
        weight_type="pair_product",
        output_savg=cfg.output_savg,
    )

def run_dd_cross_xyz(x1, y1, z1, w1,
                     x2, y2, z2, w2,
                     cfg):
    return Corrfunc.mocks.DDsmu_mocks(
        autocorr=0,
        nthreads=cfg.nthreads,
        binfile=cfg.binfile,
        mumax=cfg.mu_max,
        nmubins=cfg.nmu_bins,
        X1=x1.astype(np.float64),
        Y1=y1.astype(np.float64),
        Z1=z1.astype(np.float64),
        weights1=w1.astype(np.float64),
        X2=x2.astype(np.float64),
        Y2=y2.astype(np.float64),
        Z2=z2.astype(np.float64),
        weights2=w2.astype(np.float64),
        weight_type="pair_product",
        output_savg=cfg.output_savg,
    )

def run_dd_auto_radec(ra, dec, chi, weight, cfg):
    return DDsmu_mocks(
        autocorr=1,
        cosmology=1,
        nthreads=cfg.nthreads,
        mumax=cfg.mu_max,
        nmu_bins=cfg.nmu_bins,
        binfile=cfg.binfile,
        RA1=ra.astype(np.float64),
        DEC1=dec.astype(np.float64),
        CZ1=chi.astype(np.float64),
        weights1=weight.astype(np.float64),
        weight_type="pair_product",
        is_comoving_dist=True,
        output_savg=cfg.output_savg,
    )


def run_dd_cross_radec(ra1, dec1, chi1, w1,
                       ra2, dec2, chi2, w2,
                       cfg):
    return DDsmu_mocks(
        autocorr=0,
        cosmology=1,
        nthreads=cfg.nthreads,
        mu_max=cfg.mu_max,
        nmu_bins=cfg.nmu_bins,
        binfile=cfg.binfile,
        RA1=ra1.astype(np.float64),
        DEC1=dec1.astype(np.float64),
        CZ1=chi1.astype(np.float64),
        weights1=w1.astype(np.float64),
        RA2=ra2.astype(np.float64),
        DEC2=dec2.astype(np.float64),
        CZ2=chi2.astype(np.float64),
        weights2=w2.astype(np.float64),
        weight_type="pair_product",
        is_comoving_dist=True,
        output_savg=cfg.output_savg,
    )


def run_dd_dr_rr(
    dra, ddec, dz, dw,
    rra, rdec, rz, rw,
    cosmo,
    cfg,
    ):
    if cfg.mode == "radec":
        dchi = z_to_chi(dz, cosmo)
        rchi = z_to_chi(rz, cosmo)

        DD = run_dd_auto_radec(dra, ddec, dchi, dw, cfg)
        RR = run_dd_auto_radec(rra, rdec, rchi, rw, cfg)
        DR = run_dd_cross_radec(
            dra, ddec, dchi, dw,
            rra, rdec, rchi, rw,
            cfg,
        )

    elif cfg.mode == "xyz":
        dx, dy, dzpos = radecz_to_xyz(dra, ddec, dz, cosmo)
        rx, ry, rzpos = radecz_to_xyz(rra, rdec, rz, cosmo)

        DD = run_dd_auto_xyz(dx, dy, dzpos, dw, cfg)
        RR = run_dd_auto_xyz(rx, ry, rzpos, rw, cfg)
        DR = run_dd_cross_xyz(
            dx, dy, dzpos, dw,
            rx, ry, rzpos, rw,
            cfg,
        )

    else:
        raise ValueError("mode must be 'radec' or 'xyz'")

    return DD, DR, RR