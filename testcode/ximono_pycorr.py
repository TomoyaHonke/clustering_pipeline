from astropy.io import fits
import numpy as np
import Corrfunc
from Corrfunc.mocks import DDsmu_mocks
import astropy.units as u
import pandas as pd
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler
from sklearn import cluster
from astropy.cosmology import FlatLambdaCDM

c = 299792.458

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
ns = len(binfile) - 1
s_jack = binfile                    
mu_jack = np.linspace(-1, 1, nmu_bins + 1)
n_jack = 64
s = np.sqrt(s_jack[:-1] * s_jack[1:])

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
dat_cz_1 = c * dat_zred_1

dat_ra_2 = dat_ra[d2]
dat_dec_2 = dat_dec[d2]
dat_zred_2 = dat_zred[d2]
wD_2 = wD[d2]
dat_cz_2 = c * dat_zred_2

dat_chi_1 = cosmo.comoving_distance(dat_zred_1).to(u.Mpc).value
ra_1  = np.deg2rad(dat_ra_1)
dec_1 = np.deg2rad(dat_dec_1)
dat_x_1 = dat_chi_1 * np.cos(dec_1) * np.cos(ra_1)
dat_y_1 = dat_chi_1 * np.cos(dec_1) * np.sin(ra_1)
dat_z_1 = dat_chi_1 * np.sin(dec_1)
dat_x_1 *= h
dat_y_1 *= h
dat_z_1 *= h

dat_chi_2 = cosmo.comoving_distance(dat_zred_2).to(u.Mpc).value
ra_2  = np.deg2rad(dat_ra_2)
dec_2 = np.deg2rad(dat_dec_2)
dat_x_2 = dat_chi_2 * np.cos(dec_2) * np.cos(ra_2)
dat_y_2 = dat_chi_2 * np.cos(dec_2) * np.sin(ra_2)
dat_z_2 = dat_chi_2 * np.sin(dec_2)
dat_x_2 *= h
dat_y_2 *= h
dat_z_2 *= h


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

ran_ra_2 = ran_ra[r2]
ran_dec_2 = ran_dec[r2]
ran_zred_2 = ran_zred[r2]
wR_2 = wR[r2]

ran_chi_1 = cosmo.comoving_distance(ran_zred_1).to(u.Mpc).value
ra_1  = np.deg2rad(ran_ra_1)
dec_1 = np.deg2rad(ran_dec_1)
ran_x_1 = ran_chi_1 * np.cos(dec_1) * np.cos(ra_1)
ran_y_1 = ran_chi_1 * np.cos(dec_1) * np.sin(ra_1)
ran_z_1 = ran_chi_1 * np.sin(dec_1)
ran_x_1 *= h
ran_y_1 *= h
ran_z_1 *= h

ran_chi_2 = cosmo.comoving_distance(ran_zred_2).to(u.Mpc).value
ra_2  = np.deg2rad(ran_ra_2)
dec_2 = np.deg2rad(ran_dec_2)
ran_x_2 = ran_chi_2 * np.cos(dec_2) * np.cos(ra_2)
ran_y_2 = ran_chi_2 * np.cos(dec_2) * np.sin(ra_2)
ran_z_2 = ran_chi_2 * np.sin(dec_2)
ran_x_2 *= h
ran_y_2 *= h
ran_z_2 *= h

pos_ang_rand_1 = np.column_stack([ran_ra_1, ran_dec_1])
subsampler_1 = KMeansSubsampler(
    mode='angular',        
    positions=(dat_ra_1, dat_dec_1),
    nsamples=n_jack,       
    position_type='rd',            
)

labels_data_1 = subsampler_1.label((dat_ra_1, dat_dec_1), position_type='rd')
labels_rand_1 = subsampler_1.label((ran_ra_1, ran_dec_1), position_type='rd')

res_1 = TwoPointCorrelationFunction(
    'smu',
    edges=(s_jack, mu_jack),
    data_positions1=(dat_x_1, dat_y_1, dat_z_1),
    randoms_positions1=(ran_x_1, ran_y_1, ran_z_1),
    data_weights1=wD_1,
    randoms_weights1=wR_1,
    estimator='landyszalay',
    engine='corrfunc',
    nthreads=nthreads,
    data_samples1=labels_data_1,
    randoms_samples1=labels_rand_1,
)

xi0_1 = res_1(ell=0)
xi0_jk_1 = np.array([
    res_1.realization(ii)(ell=0)
    for ii in res_1.realizations
])

N_1 = xi0_jk_1.shape[0]
cov_xi0_1 = (N_1 - 1) * np.cov(xi0_jk_1, rowvar=False, ddof=0)
xi_mono_1 = xi0_1[0]
err_xi0_1 = np.sqrt(np.diag(cov_xi0_1))

np.savez(
    "/data/honke/corr_clustering/xi/xi0_NGC_z08_11_jk_v3.npz",
    s=s,
    xi=xi_mono_1,
    cov=cov_xi0_1,
    err=err_xi0_1,
    wR_tot=np.sum(wR_1), 
    wD_tot=np.sum(wD_1), 
    region="NGC",
)


pos_ang_rand_2 = np.column_stack([ran_ra_2, ran_dec_2])
subsampler_2 = KMeansSubsampler(
    mode='angular',        
    positions=(dat_ra_2, dat_dec_2),
    nsamples=n_jack,       
    position_type='rd',            
)

labels_data_2 = subsampler_2.label((dat_ra_2, dat_dec_2), position_type='rd')
labels_rand_2 = subsampler_2.label((ran_ra_2, ran_dec_2), position_type='rd')

res_2 = TwoPointCorrelationFunction(
    'smu',
    edges=(s_jack, mu_jack),
    data_positions1=(dat_x_2, dat_y_2, dat_z_2),
    randoms_positions1=(ran_x_2, ran_y_2, ran_z_2),
    data_weights1=wD_2,
    randoms_weights1=wR_2,
    estimator='landyszalay',
    engine='corrfunc',
    nthreads=nthreads,
    data_samples1=labels_data_2,
    randoms_samples1=labels_rand_2,
)

xi0_2 = res_2(ell=0)
xi0_jk_2 = np.array([
    res_2.realization(ii)(ell=0)
    for ii in res_2.realizations
])

N_2 = xi0_jk_2.shape[0]
cov_xi0_2 = (N_2 - 1) * np.cov(xi0_jk_2, rowvar=False, ddof=0)
xi_mono_2 = xi0_2[0]
err_xi0_2 = np.sqrt(np.diag(cov_xi0_2))

np.savez(
    "/data/honke/corr_clustering/xi/xi0_NGC_z11_16_jk_v3.npz",
    s=s,
    xi=xi_mono_2,
    cov=cov_xi0_2,
    err=err_xi0_2,
    wR_tot=np.sum(wR_2), 
    wD_tot=np.sum(wD_2), 
    region="NGC",
)

#========================================================================

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
dat_cz_1 = c * dat_zred_1

dat_ra_2 = dat_ra[d2]
dat_dec_2 = dat_dec[d2]
dat_zred_2 = dat_zred[d2]
wD_2 = wD[d2]
dat_cz_2 = c * dat_zred_2

dat_chi_1 = cosmo.comoving_distance(dat_zred_1).to(u.Mpc).value
ra_1  = np.deg2rad(dat_ra_1)
dec_1 = np.deg2rad(dat_dec_1)
dat_x_1 = dat_chi_1 * np.cos(dec_1) * np.cos(ra_1)
dat_y_1 = dat_chi_1 * np.cos(dec_1) * np.sin(ra_1)
dat_z_1 = dat_chi_1 * np.sin(dec_1)
dat_x_1 *= h
dat_y_1 *= h
dat_z_1 *= h

dat_chi_2 = cosmo.comoving_distance(dat_zred_2).to(u.Mpc).value
ra_2  = np.deg2rad(dat_ra_2)
dec_2 = np.deg2rad(dat_dec_2)
dat_x_2 = dat_chi_2 * np.cos(dec_2) * np.cos(ra_2)
dat_y_2 = dat_chi_2 * np.cos(dec_2) * np.sin(ra_2)
dat_z_2 = dat_chi_2 * np.sin(dec_2)
dat_x_2 *= h
dat_y_2 *= h
dat_z_2 *= h



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

ran_ra_2 = ran_ra[r2]
ran_dec_2 = ran_dec[r2]
ran_zred_2 = ran_zred[r2]
wR_2 = wR[r2]

ran_chi_1 = cosmo.comoving_distance(ran_zred_1).to(u.Mpc).value
ra_1  = np.deg2rad(ran_ra_1)
dec_1 = np.deg2rad(ran_dec_1)
ran_x_1 = ran_chi_1 * np.cos(dec_1) * np.cos(ra_1)
ran_y_1 = ran_chi_1 * np.cos(dec_1) * np.sin(ra_1)
ran_z_1 = ran_chi_1 * np.sin(dec_1)
ran_x_1 *= h
ran_y_1 *= h
ran_z_1 *= h

ran_chi_2 = cosmo.comoving_distance(ran_zred_2).to(u.Mpc).value
ra_2  = np.deg2rad(ran_ra_2)
dec_2 = np.deg2rad(ran_dec_2)
ran_x_2 = ran_chi_2 * np.cos(dec_2) * np.cos(ra_2)
ran_y_2 = ran_chi_2 * np.cos(dec_2) * np.sin(ra_2)
ran_z_2 = ran_chi_2 * np.sin(dec_2)
ran_x_2 *= h
ran_y_2 *= h
ran_z_2 *= h


pos_ang_rand_1 = np.column_stack([ran_ra_1, ran_dec_1])
subsampler_1 = KMeansSubsampler(
    mode='angular',        
    positions=(dat_ra_1, dat_dec_1),
    nsamples=n_jack,       
    position_type='rd',            
)

labels_data_1 = subsampler_1.label((dat_ra_1, dat_dec_1), position_type='rd')
labels_rand_1 = subsampler_1.label((ran_ra_1, ran_dec_1), position_type='rd')

res_1 = TwoPointCorrelationFunction(
    'smu',
    edges=(s_jack, mu_jack),
    data_positions1=(dat_x_1, dat_y_1, dat_z_1),
    randoms_positions1=(ran_x_1, ran_y_1, ran_z_1),
    data_weights1=wD_1,
    randoms_weights1=wR_1,
    estimator='landyszalay',
    engine='corrfunc',
    nthreads=nthreads,
    data_samples1=labels_data_1,
    randoms_samples1=labels_rand_1,
)

xi0_1 = res_1(ell=0)
xi0_jk_1 = np.array([
    res_1.realization(ii)(ell=0)
    for ii in res_1.realizations
])

N_1 = xi0_jk_1.shape[0]
cov_xi0_1 = (N_1 - 1) * np.cov(xi0_jk_1, rowvar=False, ddof=0)
xi_mono_1 = xi0_1[0]
err_xi0_1 = np.sqrt(np.diag(cov_xi0_1))

np.savez(
    "/data/honke/corr_clustering/xi/xi0_SGC_z08_11_jk_v3.npz",
    s=s,
    xi=xi_mono_1,
    cov=cov_xi0_1,
    err=err_xi0_1,
    wR_tot=np.sum(wR_1), 
    wD_tot=np.sum(wD_1), 
    region="SGC",
)


pos_ang_rand_2 = np.column_stack([ran_ra_2, ran_dec_2])
subsampler_2 = KMeansSubsampler(
    mode='angular',        
    positions=(dat_ra_2, dat_dec_2),
    nsamples=n_jack,       
    position_type='rd',            
)

labels_data_2 = subsampler_2.label((dat_ra_2, dat_dec_2), position_type='rd')
labels_rand_2 = subsampler_2.label((ran_ra_2, ran_dec_2), position_type='rd')

res_2 = TwoPointCorrelationFunction(
    'smu',
    edges=(s_jack, mu_jack),
    data_positions1=(dat_x_2, dat_y_2, dat_z_2),
    randoms_positions1=(ran_x_2, ran_y_2, ran_z_2),
    data_weights1=wD_2,
    randoms_weights1=wR_2,
    estimator='landyszalay',
    engine='corrfunc',
    nthreads=nthreads,
    data_samples1=labels_data_2,
    randoms_samples1=labels_rand_2,
)

xi0_2 = res_2(ell=0)
xi0_jk_2 = np.array([
    res_2.realization(ii)(ell=0)
    for ii in res_2.realizations
])

N_2 = xi0_jk_2.shape[0]
cov_xi0_2 = (N_2 - 1) * np.cov(xi0_jk_2, rowvar=False, ddof=0)
xi_mono_2 = xi0_2[0]
err_xi0_2 = np.sqrt(np.diag(cov_xi0_2))

np.savez(
    "/data/honke/corr_clustering/xi/xi0_SGC_z11_16_jk_v3.npz",
    s=s,
    xi=xi_mono_2,
    cov=cov_xi0_2,
    err=err_xi0_2,
    wR_tot=np.sum(wR_2), 
    wD_tot=np.sum(wD_2), 
    region="SGC",
)

