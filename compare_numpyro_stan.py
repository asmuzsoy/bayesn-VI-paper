import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

"""sn_list = pd.read_csv('data/lcs/Foundation_DR1/Foundation_DR1/Foundation_DR1.LIST', names=['file'])
sn_list['sn'] = sn_list.file.apply(lambda x: x[x.rfind('_') + 1: x.rfind('.')])
meta_file = pd.read_csv('data/lcs/meta/T21_training_set_meta.txt', delim_whitespace=True)
sn_list = sn_list.merge(meta_file, left_on='sn', right_on='SNID')

T21_mus, T21_mu_errs = [], []
for i, sn in enumerate(sn_list.sn.values):
    chains_file = f'{sn}_chains.npy'
    T21_chains = np.load(f'model_files/T21_model/T21_free_tmax/{chains_file}', allow_pickle=True).item()
    T21_mus.append(T21_chains['mu'].mean())
    T21_mu_errs.append(T21_chains['mu'].std())
T21_mus, T21_mu_errs = np.array(T21_mus), np.array(T21_mu_errs)

plt.scatter(sn_list.REDSHIFT_CMB, T21_mus)
plt.show()

with open('results/foundation_fit_T21tmax/chains.pkl', 'rb') as file:
    np_chains = pickle.load(file)

np_mus, np_mu_errs = np_chains['mu'].mean(axis=(0, 1)), np_chains['mu'].std(axis=(0, 1))
plt.scatter(sn_list.REDSHIFT_CMB, np_mus)
plt.show()
fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
ax[0].errorbar(T21_mus, np_mus, xerr=T21_mu_errs, yerr=np_mu_errs, fmt='x')
ax[1].errorbar(T21_mus, np_mus - T21_mus, xerr=T21_mu_errs, yerr=np_mu_errs, fmt='x')
print(np.std(np_mus - T21_mus))
plt.subplots_adjust(hspace=0, wspace=0)
ax[0].set_ylabel('Numpyro distance modulus')
ax[1].set_ylabel('Residual')
ax[1].set_xlabel('Stan distance modulus')
plt.savefig('plots/stan_vs_numpyro_distmod.png')
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
ax[0].scatter(T21_mu_errs, np_mu_errs)
ax[1].scatter(T21_mu_errs, np_mu_errs - T21_mu_errs)
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()
"""

"""stan_chains = np.load('model_files/T21_model/chains_realn157F_alpha_4x500+500_nou_av-exp_W1_210204_175917.npy', allow_pickle=True).item()
stan_Rv, stan_Rv_err = stan_chains['RV'].mean(axis=0), stan_chains['RV'].std(axis=0)

print(stan_chains.keys())

with open('results/T21_popRv/chains.pkl', 'rb') as file:
    np_chains = pickle.load(file)

np_Rv, np_Rv_err = np_chains['Rv'].mean(axis=(0, 1)), np_chains['Rv'].std(axis=(0, 1))

plt.scatter(stan_Rv, np_Rv)
plt.errorbar(stan_Rv, np_Rv, xerr=stan_Rv_err, yerr=np_Rv_err, fmt='x')
plt.show()

print(stan_chains['muRV'].shape)
print(np_chains['mu_R'].flatten().shape)

bins = np.arange(1.8, 4.4, 0.2)
plt.hist(stan_chains['muRV'], histtype='step', label='Stan', density=True, bins=bins)
plt.hist(np_chains['mu_R'].flatten(), histtype='step', label='Numpyro', density=True, bins=bins)
plt.xlabel(r'$\mu_R$')
plt.legend()
plt.show()

bins = np.arange(0, 1.8, 0.1)
plt.hist(stan_chains['sigmaRV'], histtype='step', label='Stan', density=True, bins=bins)
plt.hist(np_chains['sigma_R'].flatten(), histtype='step', label='Numpyro', density=True, bins=bins)
plt.xlabel(r'$\sigma_R$')
plt.legend()
plt.show()"""

with open('results/T21_popRv/chains.pkl', 'rb') as file:
    np_chains = pickle.load(file)

plt.hist(np_chains['Rv'].mean(axis=(0, 1)))
plt.show()
