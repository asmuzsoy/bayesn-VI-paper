import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner


mcmc_point_estimates = {'AV':[], 'mu':[], 'theta':[]}
vi_point_estimates = {'AV':[], 'mu':[], 'theta':[]}

mcmc_uncertainties = {'AV':[], 'mu':[], 'theta':[]}
vi_uncertainties = {'AV':[], 'mu':[], 'theta':[]}


fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

num_to_plot = 200

dataset_number=23

av_stat = 'median'

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

		if var == "AV":
			if av_stat == 'median':
				vi_point_estimates[var].append(np.median(vi_samples))
				mcmc_point_estimates[var].append(np.median(mcmc_samples))
			elif av_stat == 'mode':
				mcmc_point_estimates[var].append(get_mode_from_samples(mcmc_samples))
				vi_point_estimates[var].append(get_mode_from_samples(vi_samples))

		else: # Use sample median for mu point estimate
			vi_point_estimates[var].append(np.median(vi_samples))
			mcmc_point_estimates[var].append(np.median(mcmc_samples))

		vi_uncertainties[var].append(np.std(vi_samples))
		mcmc_uncertainties[var].append(np.std(mcmc_samples))


for var in ['AV', 'mu', 'theta']:
	mcmc_point_estimates[var] = np.array(mcmc_point_estimates[var])
	vi_point_estimates[var] = np.array(vi_point_estimates[var])
	vi_uncertainties[var] = np.array(vi_uncertainties[var])
	mcmc_uncertainties[var] = np.array(mcmc_uncertainties[var])

# VI vs true subplots
alpha = 0.2
fig, ax = plt.subplots(3,3, figsize = (9,9))
mcmc_color = 'r'
vi_color = 'b'
axis_fontsize = 12

true_values = {'mu': true_mus, 'theta': true_thetas, 'AV': true_avs}
latex_version = {'mu': '$\\mu$', 'theta': '$\\theta$', 'AV': '$A_V$'}

for i, var in enumerate(['mu', 'theta', 'AV']):
	ax[0][i].plot(true_values[var], vi_point_estimates[var], '.', alpha=alpha, c=vi_color, label='VI')
	ax[0][i].plot(true_values[var], mcmc_point_estimates[var], '.', c=mcmc_color, alpha=alpha, label='MCMC')
	ax[0][i].errorbar(true_values[var], vi_point_estimates[var], yerr = vi_uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
	ax[0][i].errorbar(true_values[var], mcmc_point_estimates[var], yerr = mcmc_uncertainties[var], c=mcmc_color, alpha=alpha, linestyle='None')
	linspace_vals = np.linspace(min(true_values[var]), max(true_values[var]))
	ax[0][i].plot(linspace_vals, linspace_vals, c='k')
	ax[0][i].set_ylabel('Fit '+ latex_version[var], fontsize = axis_fontsize)
	ax[0][i].legend()

	ax[1][i].plot(true_values[var], vi_point_estimates[var] - true_values[var], '.', alpha=alpha, c = vi_color, label='VI')
	ax[1][i].plot(true_values[var], mcmc_point_estimates[var] - true_values[var], '.', c=mcmc_color, alpha=alpha, label='MCMC')
	ax[1][i].errorbar(true_values[var], vi_point_estimates[var] - true_values[var], yerr = vi_uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
	ax[1][i].errorbar(true_values[var], mcmc_point_estimates[var] - true_values[var], yerr = mcmc_uncertainties[var], c=mcmc_color, alpha=alpha, linestyle='None')
	ax[1][i].axhline(0, color = 'k')
	ax[1][i].set_ylabel('Residual ' + latex_version[var] + ' (Fit - true)', fontsize = axis_fontsize)
	ax[1][i].legend()

	ax[2][i].plot(true_values[var], vi_point_estimates[var] - mcmc_point_estimates[var], 'o', color='gray')
	# ax[2][i].errorbar(true_values[var], vi_point_estimates[var] - mcmc_point_estimates[var], yerr = np.max([vi_uncertainties[var], mcmc_uncertainties[var]]), c='gray', linestyle='None')
	
	ax[2][i].axhline(0, color = 'k')
	ax[2][i].set_ylabel('Residual ' + latex_version[var] + ' (VI - MCMC)', fontsize = axis_fontsize)
	ax[2][i].set_xlabel('True ' + latex_version[var], fontsize = axis_fontsize)

ax[0][0].annotate("$\\mu$: median", (36, 34.5), fontsize = 16)
ax[0][1].annotate("$\\theta$: median", (0,-1.5), fontsize = 16)

if av_stat == 'median':
	ax[0][2].annotate("$A_V$: median", (0.5,0.2), fontsize = 16)
elif av_stat == 'mode':
	ax[0][2].annotate("$A_V$: mode", (0.7,0.2), fontsize = 16)


for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,3)
for i, var in enumerate(['mu', 'theta', 'AV']):
	ax[i].plot(vi_uncertainties[var], mcmc_uncertainties[var], 'o')
	min_value = np.min(np.concatenate((vi_uncertainties[var], mcmc_uncertainties[var])))
	max_value = np.max(np.concatenate((vi_uncertainties[var], mcmc_uncertainties[var])))
	linspace_vals = np.linspace(min_value, max_value)
	ax[i].plot(linspace_vals, linspace_vals, 'k')

	ax[i].set_ylabel('MCMC '  + latex_version[var] + ' uncertainty', fontsize = axis_fontsize)
	ax[i].set_xlabel('VI '  + latex_version[var] + ' uncertainty', fontsize = axis_fontsize)


plt.show()