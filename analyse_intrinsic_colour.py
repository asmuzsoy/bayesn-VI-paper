import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits, ascii
from scipy.stats import pearsonr
from scipy import odr
import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})
from lmfit.models import LinearModel
# mpl.use('macosx')

def line(x, m, c):
    return m * x + c

def correlation_step_plot(df, target_param, xlabel, ylabel, split_point=None):
    param_df = df.dropna(subset=target_param)
    print(target_param, param_df.shape)
    if f'b_{target_param}' in param_df.columns:
        err = np.mean([param_df[target_param] - param_df[f'b_{target_param}'],
                       param_df[f'B_{target_param}'] - param_df[target_param]], axis=0)
        floor = err[err > 0].min()
        err = np.max([err, np.ones_like(err) * floor], axis=0)
        err = np.nan_to_num(err, nan=floor)
    else:
        err = param_df[f'{target_param}_err']
        floor = err[err > 0].min()
        err = np.max([err, np.ones_like(err) * floor], axis=0)
        err = np.nan_to_num(err, nan=floor)
    plt.figure(figsize=(12, 8))
    plt.errorbar(param_df[target_param], param_df.g_r_at_max, yerr=param_df.g_r_at_max_err, fmt='x', alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if split_point is None:
        split_point = param_df[target_param].median()
    df1, df2 = param_df[param_df[target_param] < split_point], param_df[param_df[target_param] > split_point]
    wmu1, wmu2 = (df1.g_r_at_max / df1.g_r_at_max_err).sum() / (1 / df1.g_r_at_max_err).sum(), \
                 (df2.g_r_at_max / df2.g_r_at_max_err).sum() / (1 / df2.g_r_at_max_err).sum()
    N1, N2 = df1.shape[0], df2.shape[0]
    wstd1 = np.sqrt((np.power(df1.g_r_at_max - wmu1, 2) * (1 / df1.g_r_at_max_err)).sum() / (
            ((N1 - 1) / N1) * (1 / df1.g_r_at_max_err).sum()))
    wstd2 = np.sqrt((np.power(df2.g_r_at_max - wmu2, 2) * (1 / df2.g_r_at_max_err)).sum() / (
            ((N2 - 1) / N2) * (1 / df2.g_r_at_max_err).sum()))
    means = [df1[target_param].mean(), df2[target_param].mean()]
    plt.errorbar(means, [wmu1, wmu2], yerr=[wstd1, wstd2], fmt='kx')
    plt.vlines(split_point, param_df.g_r_at_max.min(), param_df.g_r_at_max.max(), ls='--', color='k')
    p, r = pearsonr(param_df[target_param], param_df.g_r_at_max)
    print(f'{target_param}: ', p, r)

    data = odr.Data(param_df[target_param], param_df.g_r_at_max, wd=1/err, we=1/param_df.g_r_at_max_err)
    fit_odr = odr.ODR(data, model=odr.unilinear)
    result = fit_odr.run()
    m, c = result.beta
    merr, cerr = result.sd_beta
    print(f'{target_param}: ', m, merr, m / merr)
    plot_x = np.linspace(param_df[target_param].min(), param_df[target_param].max(), 3)
    plot_y = m * plot_x + c
    plt.plot(plot_x, plot_y, color='b')
    plt.plot(plot_x, (m + merr) * plot_x + c - cerr, color='b', ls='-.')
    plt.plot(plot_x, (m - merr) * plot_x + c + cerr, color='b', ls='-.')



def main():
    # Load host properties-------------------
    #hdu = fits.open('data/host/J_ApJ_867_108_localsn.dat.fits')
    #host_data = hdu[1].data
    #df = pd.DataFrame.from_records(host_data)
    #df['SN'] = df.SN.apply(lambda x: x.rstrip())

    sn_list = pd.read_csv('data/LCs/Foundation/Foundation_DR1/Foundation_DR1.LIST', names=['file'])
    sn_list['sn'] = sn_list.file.apply(lambda x: x[x.rfind('_') + 1: x.rfind('.')])
    meta_file = pd.read_csv('data/LCs/meta/T21_training_set_meta.txt', delim_whitespace=True)
    sn_list = sn_list.merge(meta_file, left_on='sn', right_on='SNID')
    #df = sn_list.merge(df, how='left', left_on='sn', right_on='SN')
    #df = df.replace(-99.0, np.nan)

    host_data = pd.read_csv('data/host/GPC1v3_hosts.txt', delim_whitespace=True)
    host_data = host_data.replace('-', np.nan).replace('None', np.nan)
    for col in host_data.columns:
        if col not in ['ID', 'ra', 'dec', 'hostra', 'hostdec']:
            host_data[col] = host_data[col].astype(float)
    df = sn_list.merge(host_data, how='left', left_on='sn', right_on='ID')
    df['host_g-r'] = df.PS1gMag_local - df.PS1rMag_local
    df['host_g-r_err'] = np.sqrt(np.power(df.PS1gMagErr_local, 2) + np.power(df.PS1rMagErr_local, 2))
    df['host_u-r'] = df.SDSSuMag_local - df.SDSSrMag_local
    df['host_u-r_err'] = np.sqrt(np.power(df.SDSSuMagErr_local, 2) + np.power(df.SDSSuMagErr_local, 2))
    print(df[['sn', 'ID']])

    hres = np.load(os.path.join('results', 'foundation_fit_T21', 'hres.npy'))
    #df['Hres'] = hres[1, :]
    #df['e_Hres'] = hres[2, :]
    df['theta'] = hres[3, :]
    df['theta_err'] = hres[4, :]
    df['Av'] = hres[5, :]
    df['Av_err'] = hres[6, :]

    mags = np.load(os.path.join('results', 'foundation_fit_T21', 'rf_mags.npy'))
    colours = np.zeros((mags.shape[0], mags.shape[1] - 1, *mags.shape[2:]))
    for i in range(colours.shape[1]):
        colours[:, i, ...] = mags[:, i, ...] - mags[:, i + 1, ...]
    c, cerr = colours.mean(axis=0), colours.std(axis=0)

    eps0_mags = np.load(os.path.join('results', 'foundation_fit_T21', 'rf_mags_eps0.npy'))
    eps0colours = np.zeros((eps0_mags.shape[0], eps0_mags.shape[1] - 1, *eps0_mags.shape[2:]))
    for i in range(eps0colours.shape[1]):
        eps0colours[:, i, ...] = eps0_mags[:, i, ...] - eps0_mags[:, i + 1, ...]
    eps0c, eps0cerr = eps0colours.mean(axis=0), eps0colours.std(axis=0)
    g_r_at_max, g_r_at_max_err = c[0, 2, :], cerr[0, 2, :]
    c_eps = c - eps0c
    c_eps_err = np.sqrt(cerr * cerr + eps0c * eps0cerr)

    g_r_at_max, g_r_at_max_err = c_eps[0, 2, :], c_eps_err[0, 2, :]

    df['g_r_at_max'] = g_r_at_max
    df['g_r_at_max_err'] = g_r_at_max_err

    correlation_step_plot(df, 'host_g-r', r'Local PS1 g-r colour', 'Intrinsic g-r SN colour at peak')
    plt.savefig('plots/intrinsic_g-r_vs_host_g-r.png')
    plt.show()

    correlation_step_plot(df, 'host_u-r', r'Local SDSS u-r', 'Intrinsic g-r SN colour at peak')
    plt.savefig('plots/intrinsic_g-r_vs_host_u-r.png')
    plt.show()

    return
    correlation_step_plot(df, 'Massloc', r'$\log_{10}$(Local mass)', 'Intrinsic g-r SN colour at peak')
    plt.savefig('plots/intrinsic_g-r_vs_local_mass.png')
    plt.show()

    correlation_step_plot(df, '(u-g)loc', r'Local u-g colour', 'Intrinsic g-r SN colour at peak')
    plt.savefig('plots/intrinsic_g-r_vs_local_colour.png')
    plt.show()

    correlation_step_plot(df, 'SFRloc', r'$\log_{10}$(Local SFR)', 'Intrinsic g-r SN colour at peak')
    plt.savefig('plots/intrinsic_g-r_vs_local_SFR.png')
    plt.show()

    correlation_step_plot(df, 'Hres', r'Hubble residual', 'Intrinsic g-r SN colour at peak')
    plt.savefig('plots/intrinsic_g-r_vs_Hres.png')
    plt.show()


main()
