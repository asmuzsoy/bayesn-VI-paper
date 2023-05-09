import numpy as np
import matplotlib.pyplot as plt
import pickle

from bayesn_model import SEDmodel

model = SEDmodel()

with open('data/lcs/pickles/YSEwZTF_Foundation/dataset_flux.pkl', 'rb') as file:
    data = pickle.load(file)

with open('results/YSEwZTF+Foundation_fit/chains.pkl', 'rb') as file:
    chains = pickle.load(file)

z = data[-5, 0, :]

mu, mu_err = chains['mu'].mean(axis=(0, 1)), chains['mu'].std(axis=(0, 1))
model_mu = model.cosmo.distmod(z).value

hres_foundation = mu[-157:] - model_mu[-157:]
hres_yse = mu[:-157] - model_mu[:-157]
hres_yse_cut = hres_yse[hres_yse < 1]

print(f'Foundation: {np.std(hres_foundation)}')
print(f'YSE: {np.std(hres_yse)}')
print(f'YSE Cut: {np.std(hres_yse_cut)}')

z_plot = np.linspace(z.min(), z.max(), 100)
mu_plot = model.cosmo.distmod(z_plot).value

fig, ax = plt.subplots(2, sharex=True, figsize=(12, 16))
ax[0].plot(z_plot, mu_plot, ls='--', color='k')
ax[0].errorbar(z[:-157], mu[:-157], mu_err[:-157], fmt='bx', label='YSE')
ax[0].errorbar(z[-157:], mu[-157:], mu_err[-157:], fmt='rx', label='Foundation')
ax[1].errorbar(z[:-157], mu[:-157] - model_mu[:-157], mu_err[:-157], fmt='bx', label='YSE')
ax[1].errorbar(z[-157:], mu[-157:] - model_mu[-157:], mu_err[-157:], fmt='rx', label='Foundation')
ax[0].legend()
fig.supxlabel('Redshift')
ax[1].set_ylabel(r'$\mu$')
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()
