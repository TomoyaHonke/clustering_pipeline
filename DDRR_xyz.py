from astropy.io import fits
import numpy as np
import Corrfunc
from Corrfunc.mocks import DDsmu_mocks
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pandas as pd
import time

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
h = cosmo.h
dat_x_1 *= h
dat_y_1 *= h
dat_z_1 *= h

dat_chi_2 = cosmo.comoving_distance(dat_zred_2).to(u.Mpc).value
ra_2  = np.deg2rad(dat_ra_2)
dec_2 = np.deg2rad(dat_dec_2)
dat_x_2 = dat_chi_2 * np.cos(dec_2) * np.cos(ra_2)
dat_y_2 = dat_chi_2 * np.cos(dec_2) * np.sin(ra_2)
dat_z_2 = dat_chi_2 * np.sin(dec_2)
h = cosmo.h
dat_x_2 *= h
dat_y_2 *= h
dat_z_2 *= h


DD_1 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_1,
    dat_y_1,
    dat_z_1,
    weights1=wD_1,
    weight_type="pair_product",
)

DD_2 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_2,
    dat_y_2,
    dat_z_2,
    weights1=wD_2,
    weight_type="pair_product",
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

print("[info] start RR (18 randoms)")
t0 = time.time()

RR_1 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    ran_x_1,
    ran_y_1,
    ran_z_1,
    weights1=wR_1,
    weight_type="pair_product",
)
print(f"[info] RR_08_11 done: {time.time()-t0:.1f} s")

t0 = time.time()

RR_2 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    ran_x_2,
    ran_y_2,
    ran_z_2,
    weights1=wR_2,
    weight_type="pair_product",
)

print(f"[info] RR_11_16 done: {time.time()-t0:.1f} s")

print("[info] start DR")

t0 = time.time()

DR_1 = Corrfunc.mocks.DDsmu_mocks(
    0,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_1,
    dat_y_1,
    dat_z_1,
    wD_1,
    ran_x_1,
    ran_y_1,
    ran_z_1,
    wR_1,
    weight_type="pair_product",
)
print(f"[info] DR_08_11 done: {time.time()-t0:.1f} s")

t0 = time.time()

DR_2 = Corrfunc.mocks.DDsmu_mocks(
    0,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_2,
    dat_y_2,
    dat_z_2,
    wD_2,
    ran_x_2,
    ran_y_2,
    ran_z_2,
    wR_2,
    weight_type="pair_product",
)
print(f"[info] DR done: {time.time()-t0:.1f} s")

np.save("wD_NGC_08_11_v3.npy", wD_1)
np.save("wD_NGC_11_16_v3.npy", wD_2)
np.save("wR_NGC_08_11_v3.npy", wR_1)
np.save("wR_NGC_11_16_v3.npy", wR_2)

np.save("DD_NGC_08_11_v3.npy", DD_1)
np.save("DR_NGC_08_11_v3.npy", DR_1)
np.save("RR_NGC_08_11_v3.npy", RR_1)

np.save("DD_NGC_11_16_v3.npy", DD_2)
np.save("DR_NGC_11_16_v3.npy", DR_2)
np.save("RR_NGC_11_16_v3.npy", RR_2)

s = np.sqrt(binfile[:-1] * binfile[1:])

DDw_1 = DD_1["npairs"] * DD_1["weightavg"]
DRw_1 = DR_1["npairs"] * DR_1["weightavg"]
RRw_1 = RR_1["npairs"] * RR_1["weightavg"]

SD_1 = np.sum(wD_1)
SR_1 = np.sum(wR_1)   
DDn_1 = DDw_1 / (SD_1 * SD_1)
RRn_1 = RRw_1 / (SR_1 * SR_1)
DRn_1 = DRw_1 / (SD_1 * SR_1)

xi_smu_1 = (DDn_1 - 2*DRn_1 + RRn_1) / RRn_1

xi_smu_2d_1 = xi_smu_1.reshape(ns, nmu_bins)
xi_mono_1 = xi_smu_2d_1.mean(axis=1)

DDw_2 = DD_2["npairs"] * DD_2["weightavg"]
DRw_2 = DR_2["npairs"] * DR_2["weightavg"]
RRw_2 = RR_2["npairs"] * RR_2["weightavg"]

SD_2 = np.sum(wD_2)
SR_2 = np.sum(wR_2)   
DDn_2 = DDw_2 / (SD_2 * SD_2)
RRn_2 = RRw_2 / (SR_2 * SR_2)
DRn_2 = DRw_2 / (SD_2 * SR_2)

xi_smu_2 = (DDn_2 - 2*DRn_2 + RRn_2) / RRn_2

xi_smu_2d_2 = xi_smu_2.reshape(ns, nmu_bins)
xi_mono_2 = xi_smu_2d_2.mean(axis=1)

np.save("xi_smu_2d_NGC_08_11_v3.npy", xi_smu_2d_1)
np.save("xi_smu_2d_NGC_11_16_v3.npy", xi_smu_2d_2)
# np.save("s_bins_NGC.npy", s)

rows = []

for i_s, s_val in enumerate(s):
    for i_mu in range(nmu_bins):
        mu_center = (i_mu + 0.5) / nmu_bins * mu_max
        rows.append([
            s_val,
            mu_center,
            xi_smu_2d_1[i_s, i_mu]
        ])

df = pd.DataFrame(
    rows,
    columns=["s_hMpc", "mu", "xi"]
)

df.to_csv("xi_smu_NGC_08_11_v3.csv", index=False)
print("[info] output -> xi_smu_NGC_08_11.csv")

df = pd.DataFrame({
    "s_hMpc": s,
    "xi_mono": xi_mono_1,
})

df.to_csv("xi_mono_NGC_08_11_v3.csv", index=False)
print("[info] output -> xi_mono_NGC_08_11.csv")

rows = []

for i_s, s_val in enumerate(s):
    for i_mu in range(nmu_bins):
        mu_center = (i_mu + 0.5) / nmu_bins * mu_max
        rows.append([
            s_val,
            mu_center,
            xi_smu_2d_2[i_s, i_mu]
        ])

df = pd.DataFrame(
    rows,
    columns=["s_hMpc", "mu", "xi"]
)

df.to_csv("xi_smu_NGC_11_16_v3.csv", index=False)
print("[info] output -> xi_smu_NGC_11_16.csv")

df = pd.DataFrame({
    "s_hMpc": s,
    "xi_mono": xi_mono_2,
})

df.to_csv("xi_mono_NGC_11_16_v3.csv", index=False)
print("[info] output -> xi_mono_NGC_11_16.csv")

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
h = cosmo.h
dat_x_1 *= h
dat_y_1 *= h
dat_z_1 *= h

dat_chi_2 = cosmo.comoving_distance(dat_zred_2).to(u.Mpc).value
ra_2  = np.deg2rad(dat_ra_2)
dec_2 = np.deg2rad(dat_dec_2)
dat_x_2 = dat_chi_2 * np.cos(dec_2) * np.cos(ra_2)
dat_y_2 = dat_chi_2 * np.cos(dec_2) * np.sin(ra_2)
dat_z_2 = dat_chi_2 * np.sin(dec_2)
h = cosmo.h
dat_x_2 *= h
dat_y_2 *= h
dat_z_2 *= h


DD_1 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_1,
    dat_y_1,
    dat_z_1,
    weights1=wD_1,
    weight_type="pair_product",
)

DD_2 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_2,
    dat_y_2,
    dat_z_2,
    weights1=wD_2,
    weight_type="pair_product",
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

print("[info] start RR (18 randoms)")
t0 = time.time()

RR_1 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    ran_x_1,
    ran_y_1,
    ran_z_1,
    weights1=wR_1,
    weight_type="pair_product",
)
print(f"[info] RR_08_11 done: {time.time()-t0:.1f} s")

t0 = time.time()

RR_2 = Corrfunc.mocks.DDsmu_mocks(
    1,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    ran_x_2,
    ran_y_2,
    ran_z_2,
    weights1=wR_2,
    weight_type="pair_product",
)

print(f"[info] RR_11_16 done: {time.time()-t0:.1f} s")

print("[info] start DR")

t0 = time.time()

DR_1 = Corrfunc.mocks.DDsmu_mocks(
    0,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_1,
    dat_y_1,
    dat_z_1,
    wD_1,
    ran_x_1,
    ran_y_1,
    ran_z_1,
    wR_1,
    weight_type="pair_product",
)
print(f"[info] DR_08_11 done: {time.time()-t0:.1f} s")

t0 = time.time()

DR_2 = Corrfunc.mocks.DDsmu_mocks(
    0,
    nthreads,
    binfile,
    mu_max,
    nmu_bins,
    dat_x_2,
    dat_y_2,
    dat_z_2,
    wD_2,
    ran_x_2,
    ran_y_2,
    ran_z_2,
    wR_2,
    weight_type="pair_product",
)
print(f"[info] DR done: {time.time()-t0:.1f} s")

np.save("wD_SGC_08_11_v3.npy", wD_1)
np.save("wD_SGC_11_16_v3.npy", wD_2)
np.save("wR_SGC_08_11_v3.npy", wR_1)
np.save("wR_SGC_11_16_v3.npy", wR_2)

np.save("DD_SGC_08_11_v3.npy", DD_1)
np.save("DR_SGC_08_11_v3.npy", DR_1)
np.save("RR_SGC_08_11_v3.npy", RR_1)

np.save("DD_SGC_11_16_v3.npy", DD_2)
np.save("DR_SGC_11_16_v3.npy", DR_2)
np.save("RR_SGC_11_16_v3.npy", RR_2)

s = np.sqrt(binfile[:-1] * binfile[1:])

DDw_1 = DD_1["npairs"] * DD_1["weightavg"]
DRw_1 = DR_1["npairs"] * DR_1["weightavg"]
RRw_1 = RR_1["npairs"] * RR_1["weightavg"]

SD_1 = np.sum(wD_1)
SR_1 = np.sum(wR_1)   
DDn_1 = DDw_1 / (SD_1 * SD_1)
RRn_1 = RRw_1 / (SR_1 * SR_1)
DRn_1 = DRw_1 / (SD_1 * SR_1)

xi_smu_1 = (DDn_1 - 2*DRn_1 + RRn_1) / RRn_1

xi_smu_2d_1 = xi_smu_1.reshape(ns, nmu_bins)
xi_mono_1 = xi_smu_2d_1.mean(axis=1)

DDw_2 = DD_2["npairs"] * DD_2["weightavg"]
DRw_2 = DR_2["npairs"] * DR_2["weightavg"]
RRw_2 = RR_2["npairs"] * RR_2["weightavg"]

SD_2 = np.sum(wD_2)
SR_2 = np.sum(wR_2)   
DDn_2 = DDw_2 / (SD_2 * SD_2)
RRn_2 = RRw_2 / (SR_2 * SR_2)
DRn_2 = DRw_2 / (SD_2 * SR_2)

xi_smu_2 = (DDn_2 - 2*DRn_2 + RRn_2) / RRn_2

xi_smu_2d_2 = xi_smu_2.reshape(ns, nmu_bins)
xi_mono_2 = xi_smu_2d_2.mean(axis=1)

np.save("xi_smu_2d_SGC_08_11_v3.npy", xi_smu_2d_1)
np.save("xi_smu_2d_SGC_11_16_v3.npy", xi_smu_2d_2)
# np.save("s_bins_SGC.npy", s)

rows = []

for i_s, s_val in enumerate(s):
    for i_mu in range(nmu_bins):
        mu_center = (i_mu + 0.5) / nmu_bins * mu_max
        rows.append([
            s_val,
            mu_center,
            xi_smu_2d_1[i_s, i_mu]
        ])

df = pd.DataFrame(
    rows,
    columns=["s_hMpc", "mu", "xi"]
)

df.to_csv("xi_smu_SGC_08_11_v3.csv", index=False)
print("[info] output -> xi_smu_SGC_08_11.csv")

df = pd.DataFrame({
    "s_hMpc": s,
    "xi_mono": xi_mono_1,
})

df.to_csv("xi_mono_SGC_08_11_v3.csv", index=False)
print("[info] output -> xi_mono_SGC_08_11.csv")

rows = []

for i_s, s_val in enumerate(s):
    for i_mu in range(nmu_bins):
        mu_center = (i_mu + 0.5) / nmu_bins * mu_max
        rows.append([
            s_val,
            mu_center,
            xi_smu_2d_2[i_s, i_mu]
        ])

df = pd.DataFrame(
    rows,
    columns=["s_hMpc", "mu", "xi"]
)

df.to_csv("xi_smu_SGC_11_16_v3.csv", index=False)
print("[info] output -> xi_smu_SGC_11_16.csv")

df = pd.DataFrame({
    "s_hMpc": s,
    "xi_mono": xi_mono_2,
})

df.to_csv("xi_mono_SGC_11_16_v3.csv", index=False)
print("[info] output -> xi_mono_SGC_11_16.csv")