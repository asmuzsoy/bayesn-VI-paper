import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner


mcmc_point_estimates = {'AV':[], 'mu':[], 'theta':[]}
# vi_medians = {'AV':[], 'mu':[], 'theta':[]}
vi_point_estimates = {'AV':[], 'mu':[], 'theta':[]}

fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

mu_sample_diffs = []
av_samples = []
av_vi_samples = []

mu_samples = []
ds_samples = []
delm_samples = []

ds_sample_diffs = []
ds_mcmc_sample_diffs = []

ds_means = []
ds_errs = []

num_to_plot = 200

dataset_number=20

title_str = ""

if dataset_number == 6:
	title_str = "epsilons off for sims, on for fitting"
if dataset_number == 5 or dataset_number == 7:
	title_str = "no epsilons for sim/fitting"
if dataset_number == 4:
	title_str = "using epsilons for both sims and fitting"

true_avs = np.loadtxt("sim_population_AV_" + str(dataset_number) + ".txt")[:num_to_plot]
true_thetas = np.loadtxt("sim_population_theta_" + str(dataset_number) + ".txt")[:num_to_plot]
true_z = np.loadtxt("sim_population_z_" + str(dataset_number) + ".txt")[:num_to_plot]
true_mus = np.array([cosmo.distmod(z).value for z in true_z])

def get_mode_from_samples(samples):
	hist, bin_edges = np.histogram(samples, bins=50)
	max_index = np.argmax(hist)
	mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
	return mode


for i in range(num_to_plot):
	dataset = 'sim_population_' + str(dataset_number) + '/' + str(i)
	with (open("results/" + dataset + "_vi/chains.pkl", "rb")) as openfile:
		vi_objects = pickle.load(openfile)

	with (open("results/" + dataset + "_mcmc/chains.pkl", "rb")) as openfile:
		mcmc_objects = pickle.load(openfile)

	vi_params = np.load("results/" + dataset + "_vi/vi_params.npz")


	for var in ['AV', 'mu', 'theta']:
		mcmc_samples = mcmc_objects[var].reshape((1000,))
		vi_samples = np.squeeze(vi_objects[var])

		if var == "AV": # Use sample mode for AV point estimate
			mcmc_point_estimates[var].append(get_mode_from_samples(mcmc_samples))
			vi_point_estimates[var].append(get_mode_from_samples(vi_samples)) # mu means mean here
			av_samples.append(mcmc_samples)

		elif var == 'theta': # Use sample median for theta point estimate
			vi_point_estimates[var].append(np.median(vi_samples))
			mcmc_point_estimates[var].append(np.median(mcmc_samples))

		else: # Use sample median for mu point estimate
			vi_point_estimates[var].append(np.median(vi_samples))
			mcmc_point_estimates[var].append(np.median(mcmc_samples))
			mu_sample_diffs.append(vi_samples - mcmc_samples)	
			ds_sample_diffs.append(vi_params['mu'][-1][0] - np.median(vi_samples))
			ds_mcmc_sample_diffs.append(vi_params['mu'][-1][0] - mcmc_samples)
			ds_means.append(vi_params['mu'][-1][0])
			ds_errs.append(vi_params['cov'][-1][-1])

	mu_samples.append(np.squeeze(vi_objects['mu']))
	ds_samples.append(np.squeeze(vi_objects['Ds']))
	delm_samples.append(np.squeeze(vi_objects['delM']))
	av_vi_samples.append(np.squeeze(vi_objects['AV']))

mu_sample_diffs = np.array(mu_sample_diffs)
ds_mcmc_sample_diffs = np.array(ds_mcmc_sample_diffs)
delm_samples = np.array(delm_samples)
av_vi_samples = np.array(av_vi_samples)

ds_means = np.array(ds_means)
ds_errs = np.array(ds_errs)

av_samples = np.array(av_samples)

for var in ['AV', 'mu', 'theta']:
	mcmc_point_estimates[var] = np.array(mcmc_point_estimates[var])
	vi_point_estimates[var] = np.array(vi_point_estimates[var])

# VI vs true subplots
alpha = 0.2
fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(true_mus, vi_point_estimates['mu'], 'o', alpha=alpha)
ax[0][0].plot(np.linspace(min(true_mus), max(true_mus)),np.linspace(min(true_mus), max(true_mus)))
ax[0][0].set_ylabel('VI $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(true_mus, vi_point_estimates['mu'] - true_mus, 'o', alpha=alpha)
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (VI - true)', fontsize = 16)
ax[1][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[0][1].plot(true_thetas, vi_point_estimates['theta'], 'o', alpha=alpha)
ax[0][1].plot(np.linspace(min(true_thetas), max(true_thetas)),np.linspace(min(true_thetas), max(true_thetas)))
ax[0][1].set_ylabel('VI $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(true_thetas, vi_point_estimates['theta'] - true_thetas, 'o', alpha=alpha)
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (VI - true)', fontsize = 16)
ax[1][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[0][2].plot(true_avs, vi_point_estimates['AV'], 'o', alpha=alpha)
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('VI $A_V$', fontsize = 16)
ax[0][2].set_xlim(0,1)
ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(true_avs, vi_point_estimates['AV'] - true_avs, 'o', label='VI', alpha=alpha)
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (VI - true)', fontsize = 16)
ax[1][2].set_xlabel('True $A_V$', fontsize = 16)
# ax[1][2].set_xlim(0,1)

# for axis in ax.flatten():
#   axis.tick_params(axis='x', labelsize=12)
#   axis.tick_params(axis='y', labelsize=12)

# plt.tight_layout()
# plt.show()

# MCMC vs true subplots
# fig, ax = plt.subplots(2,3, figsize = (16,9))

mcmc_color = 'r'
ax[0][0].plot(true_mus, mcmc_point_estimates['mu'], 'o', c=mcmc_color, alpha=alpha)
ax[0][0].plot(np.linspace(min(true_mus), max(true_mus)),np.linspace(min(true_mus), max(true_mus)))
ax[0][0].set_ylabel('MCMC $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(true_mus, mcmc_point_estimates['mu'] - true_mus, 'o', c=mcmc_color, alpha=alpha)
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (MCMC - true)', fontsize = 16)
ax[1][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[0][1].plot(true_thetas, mcmc_point_estimates['theta'], 'o', c=mcmc_color, alpha=alpha)
ax[0][1].plot(np.linspace(min(true_thetas), max(true_thetas)),np.linspace(min(true_thetas), max(true_thetas)))
ax[0][1].set_ylabel('MCMC $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(true_thetas, mcmc_point_estimates['theta'] - true_thetas, 'o', c=mcmc_color, alpha=alpha)
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (MCMC - true)', fontsize = 16)
ax[1][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[0][2].plot(true_avs, mcmc_point_estimates['AV'], 'o', c=mcmc_color, alpha=alpha)
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('MCMC $A_V$', fontsize = 16)
ax[0][2].set_xlim(0,1)
ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(true_avs, mcmc_point_estimates['AV'] - true_avs, 'o', c=mcmc_color, label = 'MCMC', alpha=alpha)
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (MCMC - true)', fontsize = 16)
ax[1][2].set_xlabel('True $A_V$', fontsize = 16)
# ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.legend()
plt.show()

# VI vs MCMC subplots
fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(true_mus, vi_point_estimates['mu'], 'o')
ax[0][0].plot(np.linspace(min(true_mus), max(true_mus)),np.linspace(min(true_mus), max(true_mus)))
ax[0][0].set_ylabel('VI $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(true_mus, vi_point_estimates['mu'] - mcmc_point_estimates['mu'], 'o')
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
ax[1][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[0][1].plot(true_thetas, vi_point_estimates['theta'], 'o')
ax[0][1].plot(np.linspace(min(mcmc_point_estimates['theta']), max(mcmc_point_estimates['theta'])),np.linspace(min(mcmc_point_estimates['theta']), max(mcmc_point_estimates['theta'])))
ax[0][1].set_ylabel('VI $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(true_thetas, vi_point_estimates['theta'] - mcmc_point_estimates['theta'], 'o')
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (VI - MCMC)', fontsize = 16)
ax[1][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[0][2].plot(true_avs, vi_point_estimates['AV'], 'o')
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('VI $A_V$', fontsize = 16)
# ax[0][2].set_xlim(0,1)
# ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(true_avs, vi_point_estimates['AV'] - mcmc_point_estimates['AV'], 'o')
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (VI - MCMC)', fontsize = 16)
ax[1][2].set_xlabel('True $A_V$', fontsize = 16)
# ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

# VI vs MCMC subplots
fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(mcmc_point_estimates['mu'], vi_point_estimates['mu'], 'o')
ax[0][0].plot(np.linspace(min(mcmc_point_estimates['mu']), max(mcmc_point_estimates['mu'])),np.linspace(min(mcmc_point_estimates['mu']), max(mcmc_point_estimates['mu'])))
ax[0][0].set_ylabel('VI $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(mcmc_point_estimates['mu'], vi_point_estimates['mu'] - mcmc_point_estimates['mu'], 'o')
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
ax[1][0].set_xlabel('MCMC $\\mu$', fontsize = 16)


ax[0][1].plot(mcmc_point_estimates['theta'], vi_point_estimates['theta'], 'o')
ax[0][1].plot(np.linspace(min(mcmc_point_estimates['theta']), max(mcmc_point_estimates['theta'])),np.linspace(min(mcmc_point_estimates['theta']), max(mcmc_point_estimates['theta'])))
ax[0][1].set_ylabel('VI $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(mcmc_point_estimates['theta'], vi_point_estimates['theta'] - mcmc_point_estimates['theta'], 'o')
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (VI - MCMC)', fontsize = 16)
ax[1][1].set_xlabel('MCMC $\\theta$', fontsize = 16)

ax[0][2].plot(mcmc_point_estimates['AV'], vi_point_estimates['AV'], 'o')
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('VI $A_V$', fontsize = 16)
# ax[0][2].set_xlim(0,1)
# ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(mcmc_point_estimates['AV'], vi_point_estimates['AV'] - mcmc_point_estimates['AV'], 'o')
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (VI - MCMC)', fontsize = 16)
ax[1][2].set_xlabel('MCMC $A_V$', fontsize = 16)
# ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()


plt.plot(mcmc_point_estimates['AV'], vi_point_estimates['mu'] - mcmc_point_estimates['mu'], 'o')
median_residual_vi_mcmc = np.median(vi_point_estimates['mu'] - mcmc_point_estimates['mu'])
plt.axhline(median_residual_vi_mcmc, color='k', label='median residual = ' + str(round(median_residual_vi_mcmc, 4)))
plt.xlabel('MCMC $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
plt.legend()
plt.title(title_str)
plt.show()

plt.plot(true_avs, vi_point_estimates['mu'] - mcmc_point_estimates['mu'], 'o')
plt.axhline(median_residual_vi_mcmc, color='k', label='median residual = ' + str(round(median_residual_vi_mcmc, 4)))
plt.xlabel('True $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
plt.title(title_str)
plt.legend()

plt.show()

plt.plot(true_avs, vi_point_estimates['mu'] - true_mus, 'o')
median_residual_vi_true = np.median(vi_point_estimates['mu'] - true_mus)
plt.axhline(median_residual_vi_true, color='k', label='median residual = ' + str(round(median_residual_vi_true, 4)))
plt.xlabel('True $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (VI - true)', fontsize = 16)
plt.title(title_str)
plt.legend()
plt.show()

plt.plot(true_avs, mcmc_point_estimates['mu'] - true_mus, 'o')
median_residual_mcmc_true = np.median(mcmc_point_estimates['mu'] - true_mus)
plt.axhline(median_residual_mcmc_true, color='k', label='median residual = ' + str(round(median_residual_mcmc_true, 4)))
plt.xlabel('True $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (MCMC - true)', fontsize = 16)
plt.title(title_str)
plt.legend()
plt.show()

av_residuals = vi_point_estimates['AV'] - mcmc_point_estimates['AV']
for i, residual in enumerate(av_residuals):
	if abs(mcmc_point_estimates['AV'][i]) > 0.75:
		print(i, mcmc_point_estimates['AV'][i], residual)

plt.scatter(mcmc_point_estimates['AV'] - true_avs, vi_point_estimates['AV'] - true_avs, c = true_avs)
plt.plot(np.linspace(-0.3,0.3), np.linspace(-0.3,0.3), color='goldenrod')
cbar = plt.colorbar()
cbar.set_label("True AV")
plt.xlabel("MCMC AV Residuals (MCMC - true)")
plt.ylabel("VI AV Residuals (VI - true)")
plt.show()