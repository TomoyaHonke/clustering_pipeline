from astropy.cosmology import FlatLambdaCDM

def desi_cosmology():
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
    return cosmo