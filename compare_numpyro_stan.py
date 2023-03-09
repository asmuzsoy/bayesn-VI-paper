import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

sn_list = pd.read_csv('data/lcs/Foundation_DR1/Foundation_DR1/Foundation_DR1.LIST', names=['file'])
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

with open('results/foundation_fit_T21tmax/chains.pkl', 'rb') as file:
    np_chains = pickle.load(file)

np_mus, np_mu_errs = np_chains['mu'].mean(axis=(0, 1)), np_chains['mu'].std(axis=(0, 1))
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
