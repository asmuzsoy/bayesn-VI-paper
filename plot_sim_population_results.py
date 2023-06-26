import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner


mcmc_medians = {'AV':[], 'mu':[], 'theta':[]}
# vi_medians = {'AV':[], 'mu':[], 'theta':[]}
vi_modes = {'AV':[], 'mu':[], 'theta':[]}


num_to_plot = 100

fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)


true_avs = np.loadtxt("sim_population_AV.txt")[:num_to_plot]
true_thetas = np.loadtxt("sim_population_theta.txt")[:num_to_plot]
true_z = np.loadtxt("sim_population_z.txt")[:num_to_plot]
true_mus = np.array([cosmo.distmod(z).value for z in true_z])


for i in range(num_to_plot):
	dataset = 'sim_population/' + str(i)
	with (open("results/" + dataset + "_vi/chains.pkl", "rb")) as openfile:
		vi_objects = pickle.load(openfile)

	with (open("results/" + dataset + "_mcmc/chains.pkl", "rb")) as openfile:
		mcmc_objects = pickle.load(openfile)

	for var in ['AV', 'mu', 'theta']:
		mcmc_samples = mcmc_objects[var].reshape((1000,))
		mcmc_medians[var].append(np.median(mcmc_samples))
		if var == "AV":
			vi_samples = np.squeeze(vi_objects[var])
			hist, bin_edges = np.histogram(vi_samples, bins=50)
			max_index = np.argmax(hist)
			mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
			vi_modes[var].append(mode)

			plt.hist(vi_samples, histtype='step',  color='k', label='VI')
			plt.hist(mcmc_samples, histtype='step', color='r', label='MCMC')
			plt.axvline(mode, linestyle='dashed', color='k', label='VI mode')
			plt.axvline(np.median(mcmc_samples), linestyle='dashed', color='r', label='MCMC median')
			plt.axvline(true_avs[i], linestyle='dashed', color='tab:green', label='True value')

			plt.legend()
			plt.show()
		else:
			vi_modes[var].append(np.median(np.squeeze(vi_objects[var])))





for var in ['AV', 'mu', 'theta']:
	mcmc_medians[var] = np.array(mcmc_medians[var])
	vi_modes[var] = np.array(vi_modes[var])




fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(true_mus, vi_modes['mu'], 'o')
ax[0][0].plot(np.linspace(min(true_mus), max(true_mus)),np.linspace(min(true_mus), max(true_mus)))
ax[0][0].set_ylabel('Fit $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(true_mus, vi_modes['mu'] - true_mus, 'o')
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (fit - true)', fontsize = 16)
ax[1][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[0][1].plot(true_thetas, vi_modes['theta'], 'o')
ax[0][1].plot(np.linspace(min(true_thetas), max(true_thetas)),np.linspace(min(true_thetas), max(true_thetas)))
ax[0][1].set_ylabel('Fit $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(true_thetas, vi_modes['theta'] - true_thetas, 'o')
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (fit - true)', fontsize = 16)
ax[1][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[0][2].plot(true_avs, vi_modes['AV'], 'o')
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('Fit $A_V$', fontsize = 16)
ax[0][2].set_xlim(0,1)
ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(true_avs, vi_modes['AV'] - true_avs, 'o')
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (fit - true)', fontsize = 16)
ax[1][2].set_xlabel('True $A_V$', fontsize = 16)
ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(mcmc_medians['mu'], vi_modes['mu'], 'o')
ax[0][0].plot(np.linspace(min(mcmc_medians['mu']), max(mcmc_medians['mu'])),np.linspace(min(mcmc_medians['mu']), max(mcmc_medians['mu'])))
ax[0][0].set_ylabel('VI $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(mcmc_medians['mu'], vi_modes['mu'] - mcmc_medians['mu'], 'o')
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
ax[1][0].set_xlabel('MCMC $\\mu$', fontsize = 16)


ax[0][1].plot(mcmc_medians['theta'], vi_modes['theta'], 'o')
ax[0][1].plot(np.linspace(min(mcmc_medians['theta']), max(mcmc_medians['theta'])),np.linspace(min(mcmc_medians['theta']), max(mcmc_medians['theta'])))
ax[0][1].set_ylabel('VI $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(mcmc_medians['theta'], vi_modes['theta'] - mcmc_medians['theta'], 'o')
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (VI - MCMC)', fontsize = 16)
ax[1][1].set_xlabel('MCMC $\\theta$', fontsize = 16)

ax[0][2].plot(mcmc_medians['AV'], vi_modes['AV'], 'o')
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('VI $A_V$', fontsize = 16)
ax[0][2].set_xlim(0,1)
ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(mcmc_medians['AV'], vi_modes['AV'] - mcmc_medians['AV'], 'o')
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (VI - MCMC)', fontsize = 16)
ax[1][2].set_xlabel('MCMC $A_V$', fontsize = 16)
ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

plt.plot(mcmc_medians['AV'], vi_modes['mu'] - mcmc_medians['mu'], 'o')
plt.xlabel('MCMC $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
plt.show()