from astropy.io import fits
import numpy as np
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u



h = 0.6736
omega_b = 0.02237
omega_cdm = 0.12
H0 = 100 * h

Ob0 = omega_b / h**2
Om0 = (omega_b + omega_cdm) / h**2

cosmo = FlatLambdaCDM(
    H0=H0,
    Om0=Om0,
    Ob0=Ob0,
    Tcmb0=2.7255,
    Neff=3.044,
    m_nu=[0.06, 0.0, 0.0]
)

nthreads = 56
mu_max = 1.0
nmu_bins = 20
binfile = np.linspace(20, 200, 46)
outdir = "/data/honke/corr_clustering/counts_radec/"

NGC_dat = "/data/honke/DESI_clu/ELG_LOPnotqso_NGC_clustering.dat.fits"
DATA = fits.getdata(NGC_dat)
dat_ra = DATA["RA"]
dat_dec = DATA["DEC"]
dat_zred = DATA["Z"]
wD = ( DATA["WEIGHT"] * DATA["WEIGHT_FKP"])

d1 = (dat_zred >= 0.8) & (dat_zred < 1.1)
d2 = (dat_zred >= 1.1) & (dat_zred < 1.6)

dat_ra_1 = dat_ra[d1]
dat_dec_1 = dat_dec[d1]
dat_zred_1 = dat_zred[d1]
wD_1 = wD[d1]
dat_chi_1 = cosmo.comoving_distance(dat_zred_1).to(u.Mpc).value * h

dat_ra_2 = dat_ra[d2]
dat_dec_2 = dat_dec[d2]
dat_zred_2 = dat_zred[d2]
wD_2 = wD[d2]
dat_chi_2 = cosmo.comoving_distance(dat_zred_2).to(u.Mpc).value * h

DD_1 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_1.astype(np.float64),
    DEC1=dat_dec_1.astype(np.float64),
    CZ1=dat_chi_1.astype(np.float64),
    weights1=wD_1.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

DD_2 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_2.astype(np.float64),
    DEC1=dat_dec_2.astype(np.float64),
    CZ1=dat_chi_2.astype(np.float64),
    weights1=wD_2.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

random_files = [
    f"/data/honke/DESI_clu/ELG_LOPnotqso_NGC_{i}_clustering.ran.fits"
    for i in range(0, 18)
]

ran_ra_all  = []
ran_dec_all = []
ran_z_all   = []
wR_all      = []

for f in random_files:
    RAN = fits.getdata(f)

    ran_ra_all.append(RAN["RA"])
    ran_dec_all.append(RAN["DEC"])
    ran_z_all.append(RAN["Z"])
    wR_all.append(RAN["WEIGHT"] * RAN["WEIGHT_FKP"])

ran_ra   = np.concatenate(ran_ra_all)
ran_dec  = np.concatenate(ran_dec_all)
ran_zred = np.concatenate(ran_z_all)
wR       = np.concatenate(wR_all)

print("[info] N_rand total =", len(ran_ra))

r1 = (ran_zred >= 0.8) & (ran_zred < 1.1)
r2 = (ran_zred >= 1.1) & (ran_zred < 1.6)

ran_ra_1 = ran_ra[r1]
ran_dec_1 = ran_dec[r1]
ran_zred_1 = ran_zred[r1]
wR_1 = wR[r1]
ran_chi_1 = cosmo.comoving_distance(ran_zred_1).to(u.Mpc).value * h

ran_ra_2 = ran_ra[r2]
ran_dec_2 = ran_dec[r2]
ran_zred_2 = ran_zred[r2]
wR_2 = wR[r2]
ran_chi_2 = cosmo.comoving_distance(ran_zred_2).to(u.Mpc).value * h

RR_1 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=ran_ra_1.astype(np.float64),
    DEC1=ran_dec_1.astype(np.float64),
    CZ1=ran_chi_1.astype(np.float64),
    weights1=wR_1.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

RR_2 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=ran_ra_2.astype(np.float64),
    DEC1=ran_dec_2.astype(np.float64),
    CZ1=ran_chi_2.astype(np.float64),
    weights1=wR_2.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

DR_1 = DDsmu_mocks(
    autocorr=0,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_1.astype(np.float64),
    DEC1=dat_dec_1.astype(np.float64),
    CZ1=dat_chi_1.astype(np.float64),
    weights1=wD_1.astype(np.float64),
    RA2=ran_ra_1.astype(np.float64),
    DEC2=ran_dec_1.astype(np.float64),
    CZ2=ran_chi_1.astype(np.float64),
    weights2=wR_1.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

DR_2 = DDsmu_mocks(
    autocorr=0,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_2.astype(np.float64),
    DEC1=dat_dec_2.astype(np.float64),
    CZ1=dat_chi_2.astype(np.float64),
    weights1=wD_2.astype(np.float64),
    RA2=ran_ra_2.astype(np.float64),
    DEC2=ran_dec_2.astype(np.float64),
    CZ2=ran_chi_2.astype(np.float64),
    weights2=wR_2.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

np.save(outdir + "wD_NGC_0811_radec.npy", wD_1)
np.save(outdir + "wD_NGC_1116_radec.npy", wD_2)
np.save(outdir + "wR_NGC_0811_radec.npy", wR_1)
np.save(outdir + "wR_NGC_1116_radec.npy", wR_2)

np.save(outdir + "DD_NGC_0811_radec.npy", DD_1)
np.save(outdir + "DR_NGC_0811_radec.npy", DR_1)
np.save(outdir + "RR_NGC_0811_radec.npy", RR_1)

np.save(outdir + "DD_NGC_1116_radec.npy", DD_2)
np.save(outdir + "DR_NGC_1116_radec.npy", DR_2)
np.save(outdir + "RR_NGC_1116_radec.npy", RR_2)

#====================================================

SGC_dat = "/data/honke/DESI_clu/ELG_LOPnotqso_SGC_clustering.dat.fits"
DATA = fits.getdata(SGC_dat)
dat_ra = DATA["RA"]
dat_dec = DATA["DEC"]
dat_zred = DATA["Z"]
wD = ( DATA["WEIGHT"] * DATA["WEIGHT_FKP"])

d1 = (dat_zred >= 0.8) & (dat_zred < 1.1)
d2 = (dat_zred >= 1.1) & (dat_zred < 1.6)

dat_ra_1 = dat_ra[d1]
dat_dec_1 = dat_dec[d1]
dat_zred_1 = dat_zred[d1]
wD_1 = wD[d1]
dat_chi_1 = cosmo.comoving_distance(dat_zred_1).to(u.Mpc).value * h

dat_ra_2 = dat_ra[d2]
dat_dec_2 = dat_dec[d2]
dat_zred_2 = dat_zred[d2]
wD_2 = wD[d2]
dat_chi_2 = cosmo.comoving_distance(dat_zred_2).to(u.Mpc).value * h

DD_1 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_1.astype(np.float64),
    DEC1=dat_dec_1.astype(np.float64),
    CZ1=dat_chi_1.astype(np.float64),
    weights1=wD_1.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

DD_2 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_2.astype(np.float64),
    DEC1=dat_dec_2.astype(np.float64),
    CZ1=dat_chi_2.astype(np.float64),
    weights1=wD_2.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

random_files = [
    f"/data/honke/DESI_clu/ELG_LOPnotqso_SGC_{i}_clustering.ran.fits"
    for i in range(0, 18)
]

ran_ra_all  = []
ran_dec_all = []
ran_z_all   = []
wR_all      = []

for f in random_files:
    RAN = fits.getdata(f)

    ran_ra_all.append(RAN["RA"])
    ran_dec_all.append(RAN["DEC"])
    ran_z_all.append(RAN["Z"])
    wR_all.append(RAN["WEIGHT"] * RAN["WEIGHT_FKP"])

ran_ra   = np.concatenate(ran_ra_all)
ran_dec  = np.concatenate(ran_dec_all)
ran_zred = np.concatenate(ran_z_all)
wR       = np.concatenate(wR_all)

print("[info] N_rand total =", len(ran_ra))

r1 = (ran_zred >= 0.8) & (ran_zred < 1.1)
r2 = (ran_zred >= 1.1) & (ran_zred < 1.6)

ran_ra_1 = ran_ra[r1]
ran_dec_1 = ran_dec[r1]
ran_zred_1 = ran_zred[r1]
wR_1 = wR[r1]
ran_chi_1 = cosmo.comoving_distance(ran_zred_1).to(u.Mpc).value * h

ran_ra_2 = ran_ra[r2]
ran_dec_2 = ran_dec[r2]
ran_zred_2 = ran_zred[r2]
wR_2 = wR[r2]
ran_chi_2 = cosmo.comoving_distance(ran_zred_2).to(u.Mpc).value * h

RR_1 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=ran_ra_1.astype(np.float64),
    DEC1=ran_dec_1.astype(np.float64),
    CZ1=ran_chi_1.astype(np.float64),
    weights1=wR_1.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

RR_2 = DDsmu_mocks(
    autocorr=1,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=ran_ra_2.astype(np.float64),
    DEC1=ran_dec_2.astype(np.float64),
    CZ1=ran_chi_2.astype(np.float64),
    weights1=wR_2.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

DR_1 = DDsmu_mocks(
    autocorr=0,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_1.astype(np.float64),
    DEC1=dat_dec_1.astype(np.float64),
    CZ1=dat_chi_1.astype(np.float64),
    weights1=wD_1.astype(np.float64),
    RA2=ran_ra_1.astype(np.float64),
    DEC2=ran_dec_1.astype(np.float64),
    CZ2=ran_chi_1.astype(np.float64),
    weights2=wR_1.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

DR_2 = DDsmu_mocks(
    autocorr=0,
    cosmology=1,
    nthreads=nthreads,
    mu_max=mu_max,
    nmu_bins=nmu_bins,
    binfile=binfile,
    RA1=dat_ra_2.astype(np.float64),
    DEC1=dat_dec_2.astype(np.float64),
    CZ1=dat_chi_2.astype(np.float64),
    weights1=wD_2.astype(np.float64),
    RA2=ran_ra_2.astype(np.float64),
    DEC2=ran_dec_2.astype(np.float64),
    CZ2=ran_chi_2.astype(np.float64),
    weights2=wR_2.astype(np.float64),
    weight_type="pair_product",
    is_comoving_dist=True,
    output_savg=True
)

np.save(outdir + "wD_SGC_0811_radec.npy", wD_1)
np.save(outdir + "wD_SGC_1116_radec.npy", wD_2)
np.save(outdir + "wR_SGC_0811_radec.npy", wR_1)
np.save(outdir + "wR_SGC_1116_radec.npy", wR_2)

np.save(outdir + "DD_SGC_0811_radec.npy", DD_1)
np.save(outdir + "DR_SGC_0811_radec.npy", DR_1)
np.save(outdir + "RR_SGC_0811_radec.npy", RR_1)

np.save(outdir + "DD_SGC_1116_radec.npy", DD_2)
np.save(outdir + "DR_SGC_1116_radec.npy", DR_2)
np.save(outdir + "RR_SGC_1116_radec.npy", RR_2)