import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner
import pandas as pd

fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

num_to_plot = 157

dataset = 'T21_training_set'

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])
sn_names = list(sn_list.sn.values)

sn_names[sn_names.index("AT2016ajl")] = 'AT2016aj'

av_stat = 'median'

title_str = ""

sn_info = pd.read_csv("data/lcs/Foundation_DR1/Foundation_DR1.FITRES.TEXT", comment='#', delim_whitespace=True)


class Result:
  def __init__(self, filename):
    self.results_dict = np.load(filename, allow_pickle=True).item()
    self.mu_samples = self.results_dict['mu'].reshape((num_to_plot, 1000,))
    self.theta_samples = self.results_dict['theta'].reshape((num_to_plot, 1000,))
    self.av_samples = self.results_dict['AV'].reshape((num_to_plot, 1000,))
    self.samples_dict = {'mu':self.mu_samples, 'theta':self.theta_samples, 'AV':self.av_samples}
    self.point_estimates = {var: np.median(self.samples_dict[var], axis=1) for var in self.samples_dict.keys()}
    self.stds = {var: np.std(self.samples_dict[var], axis=1) for var in self.samples_dict.keys()}
    self.variances = {var: np.var(self.samples_dict[var], axis=1) for var in self.samples_dict.keys()}


# need to re-run MCMC with new model -> this is done
# zltn_result = Result("foundation_vmap_zltn_122923.npy")
# mcmc_result = Result("foundation_vmap_mcmc_122923.npy")
# laplace_result = Result("foundation_vmap_laplace_122923.npy")
# multinormal_result = Result("foundation_vmap_multinormal_122923.npy")

zltn_result = Result("foundation_results/foundation_vmap_zltn_032624_samples.npy")
mcmc_result = Result("foundation_results/foundation_vmap_mcmc_032624_samples.npy")
laplace_result = Result("foundation_results/foundation_vmap_laplace_032624_samples.npy")
multinormal_result = Result("foundation_results/foundation_vmap_multinormal_032624_samples.npy")


zcmb_dict = dict(zip(sn_info.CID.values, sn_info.zCMB.values))

zcmbs = np.array([zcmb_dict[name] for name in sn_names])

def get_mode_from_samples(samples):
	hist, bin_edges = np.histogram(samples, bins=50)
	max_index = np.argmax(hist)
	mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
	return mode

# load Stephen's MCMC chains for comparison
# using names to make sure they're in the same order
stephen_point_estimates = {'mu':[], 'theta':[], 'AV':[]}
stephen_uncertainties = {'mu':[], 'theta':[], 'AV':[]}
for i in range(num_to_plot):
	s = np.load("../dist_chains_210610_135216/" + sn_names[i] + "_chains_210610_135216.npy", allow_pickle=True).item()
	for var in ['mu', 'theta', "AV"]:
		stephen_point_estimates[var].append(np.median(s[var]))
		stephen_uncertainties[var].append(np.std(s[var]))

for i in range(num_to_plot - 1):
	if (zltn_result.point_estimates['theta'][i] - mcmc_result.point_estimates['theta'][i]).any() < -1.5:
		print(i, sn_names[i])


for i in range(num_to_plot):
	if np.isnan(zltn_result.point_estimates['AV'][i]):
		print(i, sn_names[i])

best_ks = np.loadtxt("foundation_results/best_ks_032824.txt")
last_ks = np.loadtxt("foundation_results/last_ks_032824.txt")

best_samples = np.load("foundation_results/best_samples_032824.npz", allow_pickle = True)['arr_0']
last_samples = np.load("foundation_results/last_samples_032824.npz", allow_pickle = True)['arr_0']

best_mu_medians = np.array([np.median(best_samples[i]['Ds']) for i in range(num_to_plot)])
last_mu_medians = np.array([np.median(last_samples[i]['Ds']) for i in range(num_to_plot)])

best_mu_std = np.array([np.std(best_samples[i]['Ds']) for i in range(num_to_plot)])
last_mu_std = np.array([np.std(last_samples[i]['Ds']) for i in range(num_to_plot)])


fig, ax = plt.subplots()
t = ax.scatter(best_ks, last_ks, c = best_mu_medians - last_mu_medians, cmap='bwr')
linspace_vals = np.linspace(0.3, 1.3)
ax.plot(linspace_vals,linspace_vals, 'k')
cbar = fig.colorbar(t)
cbar.set_label('$\\Delta \\mu$')
ax.set_xlabel("k from best-loss parameters")
ax.set_ylabel("k from last-loss parameters")
plt.show()
fig.savefig("figures/bestvslastk.pdf", bbox_inches='tight')


plt.scatter(best_ks, last_ks, c = best_mu_medians - np.array([cosmo.distmod(z).value for z in zcmbs]), cmap='bwr')
linspace_vals = np.linspace(0.3, 1.3)
plt.plot(linspace_vals,linspace_vals, 'k')
cbar = plt.colorbar()
cbar.set_label('$D_{s, best} - \\mu(z)$')
plt.xlabel("k from best loss params")
plt.ylabel("k from last iteration params")
plt.show()


plt.scatter(best_ks, last_ks, c = best_mu_std - last_mu_std, cmap='bwr')
linspace_vals = np.linspace(0.3, 1.3)
plt.plot(linspace_vals,linspace_vals, 'k')
cbar = plt.colorbar()
cbar.set_label('$D_{s, best} - \\mu(z)$')
plt.xlabel("k from best loss params")
plt.ylabel("k from last iteration params")
plt.show()

plt.plot(best_ks - last_ks, best_mu_std - last_mu_std, 'o')
plt.xlabel("delta k (best - last)")
plt.ylabel("delta stdev (best - last)")
plt.show()

fig = plt.figure()
plt.plot(last_ks, last_mu_medians - mcmc_result.point_estimates['mu'], 'o')
plt.axvline(0.7, linestyle='dashed', color='k')
plt.xlabel("$\\hat{k}$", fontsize = 20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel("MVZLTN VI $\\mu$ - MCMC $\\mu$", fontsize = 20)
fig.savefig("figures/khat_mu_comparison.pdf", bbox_inches = 'tight')
plt.show()

plt.plot(last_ks, last_mu_medians - np.array([cosmo.distmod(z).value for z in zcmbs]), 'o')
plt.xlabel("$\hat{k}$")
plt.ylabel("VI $\mu$ - $\\mu(z)$")
plt.show()


# regularized_k = (96 * last_ks + 10*0.5) / (96 + 10)
# plt.plot(regularized_k, last_mu_medians - mcmc_result.point_estimates['mu'], 'o')
# plt.xlabel("$\hat{k}$")
# plt.ylabel("VI $\mu$ - MCMC $\mu$")
# plt.show()

# VI vs MCMC subplots
alpha = 0.2
fig, ax = plt.subplots(2,3, figsize = (9,9))
mcmc_color = 'r'
vi_color = 'b'
axis_fontsize = 12

latex_version = {'mu': '$\\mu$', 'theta': '$\\theta$', 'AV': '$A_V$'}

for i, var in enumerate(['mu', 'theta', 'AV']):
	ax[0][i].plot(mcmc_result.point_estimates[var], zltn_result.point_estimates[var], '.', alpha=alpha, c=vi_color, label='VI')
	ax[0][i].errorbar(mcmc_result.point_estimates[var], zltn_result.point_estimates[var], yerr = zltn_result.stds[var], xerr = mcmc_result.stds[var], alpha=alpha, c=vi_color, linestyle='None')
	linspace_vals = np.linspace(min(mcmc_result.point_estimates[var]), max(mcmc_result.point_estimates[var]))
	ax[0][i].plot(linspace_vals, linspace_vals, c='k')
	ax[0][i].set_ylabel('VI '+ latex_version[var], fontsize = axis_fontsize)

	ax[1][i].plot(mcmc_result.point_estimates[var], zltn_result.point_estimates[var] - mcmc_result.point_estimates[var], '.', alpha=alpha, c = vi_color, label='VI')
	ax[1][i].errorbar(mcmc_result.point_estimates[var], zltn_result.point_estimates[var] - mcmc_result.point_estimates[var], xerr = mcmc_result.stds[var], yerr = zltn_result.stds[var], alpha=alpha, c=vi_color, linestyle='None')
	ax[1][i].axhline(0, color = 'k')
	ax[1][i].set_ylabel('Residual ' + latex_version[var] + ' (VI - MCMC)', fontsize = axis_fontsize)
	ax[1][i].set_xlabel('NumPyro MCMC ' + latex_version[var], fontsize = axis_fontsize)

	# ax[2][i].plot(true_values[var], zltn_result.point_estimates[var] - mcmc_result.point_estimates[var], 'o', color='gray')
	# # ax[2][i].errorbar(true_values[var], zltn_result.point_estimates[var] - mcmc_result.point_estimates[var], yerr = np.max([zltn_result.stds[var], mcmc_result.stds[var]]), c='gray', linestyle='None')
	
	# ax[2][i].axhline(0, color = 'k')
	# ax[2][i].set_ylabel('Residual ' + latex_version[var] + ' (VI - MCMC)', fontsize = axis_fontsize)


for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()


## Plot comparison to Stephen's MCMC chains
fig, ax = plt.subplots(2,3, figsize = (9,9))

for i, var in enumerate(['mu', 'theta', 'AV']):
	ax[0][i].plot(stephen_point_estimates[var], zltn_result.point_estimates[var], '.', alpha=alpha, c=vi_color, label='VI')
	ax[0][i].errorbar(stephen_point_estimates[var], zltn_result.point_estimates[var], yerr = zltn_result.stds[var], xerr = stephen_uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
	linspace_vals = np.linspace(min(stephen_point_estimates[var]), max(stephen_point_estimates[var]))
	ax[0][i].plot(linspace_vals, linspace_vals, c='k')
	ax[0][i].set_ylabel('VI '+ latex_version[var], fontsize = axis_fontsize)

	ax[1][i].plot(stephen_point_estimates[var], zltn_result.point_estimates[var] - stephen_point_estimates[var], '.', alpha=alpha, c = vi_color, label='VI')
	ax[1][i].errorbar(stephen_point_estimates[var], zltn_result.point_estimates[var] - stephen_point_estimates[var], yerr = zltn_result.stds[var], xerr = stephen_uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
	ax[1][i].axhline(0, color = 'k')
	ax[1][i].set_ylabel('Residual ' + latex_version[var] + ' (VI - MCMC)', fontsize = axis_fontsize)
	ax[1][i].set_xlabel('Stan MCMC ' + latex_version[var], fontsize = axis_fontsize)

	# ax[2][i].plot(true_values[var], zltn_result.point_estimates[var] - mcmc_result.point_estimates[var], 'o', color='gray')
	# # ax[2][i].errorbar(true_values[var], zltn_result.point_estimates[var] - mcmc_result.point_estimates[var], yerr = np.max([zltn_result.stds[var], mcmc_result.stds[var]]), c='gray', linestyle='None')
	
	# ax[2][i].axhline(0, color = 'k')
	# ax[2][i].set_ylabel('Residual ' + latex_version[var] + ' (VI - MCMC)', fontsize = axis_fontsize)


for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()


#### plot MCMC vs VI uncertainties
fig, ax = plt.subplots(1,3)
for i, var in enumerate(['mu', 'theta', 'AV']):
	ax[i].plot(zltn_result.stds[var], mcmc_result.stds[var], 'o')
	min_value = np.min(np.concatenate((zltn_result.stds[var], mcmc_result.stds[var])))
	max_value = np.max(np.concatenate((zltn_result.stds[var], mcmc_result.stds[var])))
	linspace_vals = np.linspace(min_value, max_value)
	ax[i].plot(linspace_vals, linspace_vals, 'k')
	ax[i].set_ylabel('MCMC '  + latex_version[var] + ' uncertainty', fontsize = axis_fontsize)
	ax[i].set_xlabel('VI '  + latex_version[var] + ' uncertainty', fontsize = axis_fontsize)
plt.show()


## Plot Hubble diagram
linspace_z = np.linspace(0.01, 0.085)

cosmo_distmod_values = np.array([cosmo.distmod(z).value for z in linspace_z])

def plot_hubble_distances_and_residuals(mus, variances, z_cmbs, linspace_z = linspace_z, cosmo_distmod_values = cosmo_distmod_values):
  predictions = mus
  targets = np.array([cosmo.distmod(z).value for z in z_cmbs])
  rmse = np.sqrt(np.mean((predictions-targets)**2))
  
  fig, ax = plt.subplots(2,1, figsize = (6,9), sharex='col', gridspec_kw = {'hspace':0})
  ax[0].errorbar(z_cmbs, mus, np.sqrt(variances), linestyle = 'None', color = 'k')
  ax[0].plot(z_cmbs, mus,'o')
  ax[0].plot(linspace_z, cosmo_distmod_values, color = 'k')
  ax[0].set_xlabel("z", fontsize =20)
  ax[0].set_ylabel("$\\mu$", fontsize = 20)
  print(rmse)
  plt.text(0.07, 0.8, "Foundation DR1", fontsize = 16, horizontalalignment = 'center')
  plt.text(0.07, 0.7, "N = " + str(len(z_cmbs)), fontsize = 16, horizontalalignment = 'center')
  plt.text(0.07, 0.6, "RMSE = " + '{:.3f}'.format(round(rmse, 3)), fontsize = 16, horizontalalignment = 'center')


  sigma_pec = 150
  c = 300000
  print(np.mean(mus - np.array(targets)), "+/-", np.std(mus - np.array(targets)) / np.sqrt(len(z_cmbs)))
  residuals = mus - np.array(targets)
  print("reduced chisq:", sum(residuals**2/variances) / len(mus) - 1)

  ax[1].errorbar(z_cmbs, mus - np.array(targets), np.sqrt(variances), linestyle = 'None', color = 'k')
  ax[1].plot(z_cmbs, mus - np.array(targets), 'o')

  sigma_envelope = np.array([(5 / (z * np.log(10))) * (sigma_pec / c) for z in linspace_z])
  ax[1].plot(linspace_z, sigma_envelope, marker = 'None', linestyle = 'dashed', color = 'k')
  ax[1].plot(linspace_z, -sigma_envelope, marker = 'None', linestyle = 'dashed', color = 'k')

  ax[1].axhline(0., color = 'k')

  ax[1].set_xlabel("z", fontsize = 20)
  ax[1].set_ylabel("Hubble Residual", fontsize = 20)
  ax[0].tick_params(axis='both', labelsize=18)
  ax[1].tick_params(axis='both', labelsize=18)
  return fig

fig = plot_hubble_distances_and_residuals(zltn_result.point_estimates['mu'], zltn_result.variances['mu'], zcmbs)
plt.show()
fig.savefig("figures/hubble_diagram.pdf", bbox_inches='tight')


fig = plot_hubble_distances_and_residuals(multinormal_result.point_estimates['mu'], multinormal_result.variances['mu'], zcmbs)
plt.show()
fig.savefig("figures/hubble_diagram_multinormal.pdf", bbox_inches='tight')

fig = plot_hubble_distances_and_residuals(laplace_result.point_estimates['mu'], laplace_result.variances['mu'], zcmbs)
plt.show()
fig.savefig("figures/hubble_diagram_laplace.pdf", bbox_inches='tight')

fig = plot_hubble_distances_and_residuals(mcmc_result.point_estimates['mu'], mcmc_result.variances['mu'], zcmbs)
plt.show()
fig.savefig("figures/hubble_diagram_mcmc.pdf", bbox_inches='tight')


print(mcmc_result.stds['mu'])