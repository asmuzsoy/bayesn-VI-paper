import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import Distance
import astropy.units as u
import pickle

from bayesn_model import SEDmodel

model = SEDmodel()

with open('data/lcs/pickles/YSEwZTF_Foundation/dataset_flux.pkl', 'rb') as file:
    data = pickle.load(file)

with open('results/YSEwZTF+Foundation_fit/chains.pkl', 'rb') as file:
    chains = pickle.load(file)
sn_list = pd.read_csv('results/YSEwZTF+Foundation_fit/sn_list.txt', header=None, names=['sn'])

with open('data/lcs/pickles/YSE_DR1/dataset_mag.pkl', 'rb') as file:
    T21_data = pickle.load(file).T

with open(os.path.join('results', 'YSE_full_T21_fit', 'chains.pkl'), 'rb') as file:
    T21_chains = pickle.load(file)

df = pd.read_csv('results/YSE_full_T21_fit/sn_props.txt')
df['mu'] = T21_chains['mu'].mean(axis=(0, 1))
df['mu_err'] = T21_chains['mu'].std(axis=(0, 1))
df['theta'] = T21_chains['theta'].mean(axis=(0, 1))
df['theta_err'] = T21_chains['theta'].std(axis=(0, 1))
df['Av'] = T21_chains['AV'].mean(axis=(0, 1))
df['Av_err'] = T21_chains['AV'].std(axis=(0, 1))
df['tmax'] = T21_chains['tmax'].mean(axis=(0, 1))
df['tmax_err'] = T21_chains['tmax'].std(axis=(0, 1))
df['Hres'] = df.mu - df.muhat

print('Full', df.Hres.std())

df = df[df.Av < 1]
print('Av cut', df.Hres.std())

df = df[df.theta_err < 0.5]
print('Theta error cut', df.Hres.std())

df = df[df.tmax_err < 2]
print('tmax error cut', df.Hres.std())

plt.errorbar(df.z, df.mu, yerr=df.mu_err, fmt='x')
plt.show()

"""

sn_list['z'] = data[-5, 0, :]

sn_list['mu'] = chains['mu'].mean(axis=(0, 1))
sn_list['mu_err'] = chains['mu'].std(axis=(0, 1))
sn_list['model_mu'] = model.cosmo.distmod(sn_list.z.values).value

sn_list = sn_list.merge(T21_fit_sn_list, on='sn')

inds = T21_fit_sn_list.sn.isin(sn_list.sn.values)
sn_list['T21_mu'] = T21_chains['mu'][..., inds].mean(axis=(0, 1))
sn_list['T21_mu_err'] = T21_chains['mu'][..., inds].std(axis=(0, 1))

sn_list['Hres'] = sn_list.mu - sn_list.model_mu
sn_list['T21_Hres'] = sn_list.T21_mu - sn_list.model_mu

sn_list['Hres_sigma'] = (sn_list.mu - sn_list.model_mu) / sn_list.mu_err
sn_list['T21_Hres_sigma'] = (sn_list.T21_mu - sn_list.model_mu) / sn_list.T21_mu_err

z_plot = np.linspace(sn_list.z.min(), sn_list.z.max(), 100)
mu_plot = model.cosmo.distmod(z_plot).value

print(sn_list.Hres.std())
print(sn_list.T21_Hres.std())
print(sn_list.Hres_sigma.std())
print(sn_list.T21_Hres_sigma.std())

fig, ax = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(21, 14))
ax[0, 0].plot(z_plot, mu_plot, ls='--', color='k')
ax[0, 1].plot(z_plot, mu_plot, ls='--', color='k')
ax[0, 0].errorbar(sn_list.z, sn_list.mu, sn_list.mu_err, fmt='bx', label='YSE+Foundation')
ax[0, 1].errorbar(sn_list.z, sn_list.T21_mu, sn_list.T21_mu_err, fmt='rx', label='T21')
ax[1, 0].errorbar(sn_list.z, sn_list.mu - sn_list.model_mu, sn_list.mu_err, fmt='bx', label='YSE+Foundation')
ax[1, 1].errorbar(sn_list.z, sn_list.T21_mu - sn_list.model_mu, sn_list.T21_mu_err, fmt='rx', label='T21')
#ax[0].errorbar(z[:-157], mu[:-157], mu_err[:-157], fmt='bx', label='YSE')
#ax[0].errorbar(z[-157:], mu[-157:], mu_err[-157:], fmt='rx', label='Foundation')
#ax[1].errorbar(z[:-157], mu[:-157] - model_mu[:-157], mu_err[:-157], fmt='bx', label='YSE')
#ax[1].errorbar(z[-157:], mu[-157:] - model_mu[-157:], mu_err[-157:], fmt='rx', label='Foundation')
ax[0, 0].legend()
ax[0, 1].legend()
fig.supxlabel('Redshift')
ax[0, 0].set_ylabel(r'$\mu$')
ax[1, 0].set_ylabel('Residual')
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()
"""

