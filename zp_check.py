import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.constants as const

zp_dict = {'u_CSP': -20.204375707429655, 'g_CSP': -20.802411500300547, 'r_CSP': -21.379495865204007, 'i_CSP': -21.817867384198493, 'B_CSP': -20.514234357411148, 'V_CSP_3014': -21.051661591776902, 'V_CSP_3009': -21.068030452641757, 'V_CSP': -21.048266929525113, 'Y_WIRC': -23.162481076807367, 'J_WIRC': -23.801341143541883, 'H_WIRC': -24.8268771721727, 'K_WIRC': -25.96660289092869, 'Y_RC': -23.13309690443498, 'J_RC1': -23.824138427436353, 'J_RC2': -23.803274236168047, 'H_RC': -24.830094586250556, 'u_CSP2': -20.18815270545189, 'g_CSP2': -20.794553063166315, 'r_CSP2': -21.378236449966764, 'i_CSP2': -21.823264016663956, 'B_CSP2': -20.513854208143503, 'V_CSP2': -21.044412732420774, 'Y_RCDP': -23.14012335821222, 'J_RCDP': -23.806124669990897, 'H_RCDP': -24.821496843808696, 'u_prime': -20.160263048240484, 'r_prime': -21.379039624215253, 'i_prime': -21.81145564304667, 'J': -23.787971674429393, 'H': -24.885978567243402, 'K': -25.94739878534259, 'V': -21.107250696797433, 'I': -22.37573685665803, 'U': -20.9959503244276, 'R': -21.67924804448482, 'B': -20.493196279361502, 'Y_AND': -23.1314225998679, 'J_AND': -23.815813515978004, 'H_AND': -24.842016192458036, 'K_AND': -25.916637045317756, 'g_PS1': -20.8411200358856, 'r_PS1': -21.370327375262352, 'i_PS1': -21.793353385176655, 'z_PS1': -22.099096353168953, 'y_PS1': -22.32557706995602, 'u_DES': -20.32057676436583, 'g_DES': -20.818013896700013, 'r_DES': -21.444815554988715, 'i_DES': -21.872495903265143, 'z_DES': -22.219518728998516, 'y_DES': -22.3852702022968, 'u_LSST': -20.228208642646187, 'g_LSST': -20.805179655764622, 'r_LSST': -21.376204207093203, 'i_LSST': -21.792725883667487, 'z_LSST': -22.101960486974228, 'y_LSST': -22.348986035046995, 'J_HST': -22.890074914187842, 'H_HST': -23.341365320465602, 'Y_P': -23.11235574354868, 'J_P': -23.800891280686965, 'H_P': -24.824030979642036, 'K_P': -25.93824631479329, 'B_TNT': -20.534608823868442, 'V_TNT': -21.12457424627681, 'R_TNT': -21.583458778999457, 'I_TNT': -22.31361170076449, 'J_SWI': -23.858741646440613, 'H_SWI': -24.92550712534101, 'F300X': -19.64739172496052, 'F225W': -19.269951566702645, 'F336W': -20.036150450162744, 'F275W': -19.567126290949595, 'F390W': -20.374783977949136, 'F438W': -20.587966688435102, 'F475W': -20.801522254699847, 'F555W': -21.032433538582268, 'F625W': -21.384551110036483, 'F814W': -21.93132447376678, 'F105W': -22.52427210444864, 'F125W': -22.890074914187842, 'F140W': -23.12665134023583, 'F160W': -23.341365320465602, 'U_SWIFT': -21.09920764791584, 'B_SWIFT': -20.471217784476117, 'V_SWIFT': -21.071922460025515, 'UVW1': -20.956342809138697, 'UVW2': -20.676833757188827, 'UVM2': -20.825894043318808}

def f_lam(l):
    f = (const.c.to('AA/s').value/1e23)*((l)**-2)*10**(-48.6/2.5)*1e23
    return f

with fits.open('data/zp/alpha_lyr_stis_010.fits') as hdu:
    head = hdu[0].header
    vega_df = pd.DataFrame.from_records(hdu[1].data)
    vega_lam, vega_f = vega_df.WAVELENGTH, vega_df.FLUX

obsmode = pd.read_csv('data/SNmodel_pb_obsmode_map.txt', delim_whitespace=True)
#obsmode = obsmode[obsmode.magsys == 'vegamag']

diffs = []
for i, row in obsmode.iterrows():
    filt = row.pb
    try:
        R = np.loadtxt(f'data/{row.obsmode}')
    except:
        continue
    R[:, 1] = R[:, 1] * (R[:, 1] > 0.002 * R[:, 1].max())
    lam = R[:, 0]
    if row.magsys == 'abmag':
        zp = f_lam(lam)
    elif row.magsys == 'vegamag':
        zp = interp1d(vega_lam, vega_f, kind='cubic')(lam)
    else:
        continue
    int1 = simpson(lam * zp * R[:, 1], lam)
    int2 = simpson(lam * R[:, 1], lam)
    zp_flux = int1 / int2
    zp_mag = 2.5 * np.log10(zp_flux)
    diff = zp_mag - zp_dict[filt]
    #if np.abs(diff) > 0.01:
    print(R.shape, filt, zp_flux, zp_mag, zp_dict[filt], diff)
    #    plt.plot(R[:, 0], R[:, 1])
    #    plt.title(filt)
    #    plt.show()
