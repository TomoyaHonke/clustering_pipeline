from astropy.io import fits
import numpy as np

def load_data_catalog(data_path):
    data = fits.getdata(data_path)
    ra = data["RA"]
    dec = data["DEC"]
    zred = data["Z"]
    w = data["WEIGHT"] * data["WEIGHT_FKP"]

    return ra, dec, zred, w

def load_random_catalog(random_path):
    ra_all, dec_all, z_all, w_all = [], [], [], []
    for path in random_path:
        ran = fits.getdata(path)
        ra_all.append(ran["RA"])
        dec_all.append(ran["DEC"])
        z_all.append(ran["Z"])
        w_all.append(ran["WEIGHT"] * ran["WEIGHT_FKP"])

    return (
        np.concatenate(ra_all),
        np.concatenate(dec_all),
        np.concatenate(z_all),
        np.concatenate(w_all),
    )

def select_zbin(ra, dec, z, w, zmin=None, zmax=None):
    if zmin is None or zmax is None:
        return ra, dec, z, w

    mask = (z >= zmin) & (z < zmax)
    return ra[mask], dec[mask], z[mask], w[mask]