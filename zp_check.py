import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open('data/zp/alpha_lyr_stis_008.fits') as hdu:
    head = hdu[0].header
    df = pd.DataFrame.from_records(hdu[1].data)

R = np.loadtxt(f'data/filters/LCO/Swope/B_tel_ccd_atm_ext_1.2.dat')
R[:, 1] = R[:, 1] * (R[:, 1] > 0.002 * R[:, 1].max())
lam = R[:, 0]
vega = interp1d(df.WAVELENGTH, df.FLUX, kind='cubic')(lam)
int1 = simpson(lam * vega * R[:, 1], lam)
int2 = simpson(lam * R[:, 1], lam)
zp_flux = int1 / int2
zp_mag = -2.5 * np.log10(zp_flux)
print(zp_flux, zp_mag)
