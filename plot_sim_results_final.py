import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner
from scipy import special, stats
from decimal import Decimal

fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

num_to_plot = 1000

dataset_number=29

av_stat = 'median'

title_str = ""

true_avs = np.loadtxt("sim_population_AV_" + str(dataset_number) + ".txt")[:num_to_plot]
true_thetas = np.loadtxt("sim_population_theta_" + str(dataset_number) + ".txt")[:num_to_plot]
true_z = np.loadtxt("sim_population_z_" + str(dataset_number) + ".txt")[:num_to_plot]
true_mus = np.array([cosmo.distmod(z).value for z in true_z])

def get_mode_from_samples(samples):
  hist, bin_edges = np.histogram(samples, bins=50)
  max_index = np.argmax(hist)
  mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
  return mode

class Result:
  def __init__(self, filename, av_metric = 'median'):
    self.results_dict = np.load(filename, allow_pickle=True).item()
    self.mu_samples = self.results_dict['mu'].reshape((num_to_plot, 1000,))
    self.theta_samples = self.results_dict['theta'].reshape((num_to_plot, 1000,))
    self.av_samples = self.results_dict['AV'].reshape((num_to_plot, 1000,))
    self.samples_dict = {'mu':self.mu_samples, 'theta':self.theta_samples, 'AV':self.av_samples}
    self.point_estimates = {var: np.median(self.samples_dict[var], axis=1) for var in self.samples_dict.keys()}
    if av_metric == 'mode':
    	print("yes")
    	self.point_estimates['AV'] = np.array([get_mode_from_samples(s) for s in self.av_samples])
    self.stds = {var: np.std(self.samples_dict[var], axis=1) for var in self.samples_dict.keys()}
    self.variances = {var: np.var(self.samples_dict[var], axis=1) for var in self.samples_dict.keys()}

zltn_result = Result("sim29_vmap_zltn_022824_samples.npy", av_metric=av_stat)
mcmc_result = Result("sim29_vmap_mcmc_022824_samples.npy")
laplace_result = Result("sim29_vmap_laplace_022824_samples.npy")
multinormal_result = Result("sim29_vmap_multinormal_022824_samples.npy")

labels = ['MCMC', 'MVZLTN', 'Multivariate Normal', 'Laplace Approximation']
latex_version = {'mu': '$\\mu$', 'theta': '$\\theta_1$', 'AV': '$A_V$'}
prob_labels = {'AV': '$p_{A_V}$', "mu": '$p_\\mu$', "theta": '$p_{\\theta_1}$'}
true_values = {'AV': true_avs, "mu": true_mus, 'theta': true_thetas}

subplots_legend = np.empty((3,4), dtype=object)

num_samples_per_sn = 1000
# VSBC Figure
fig, ax = plt.subplots(3,4, figsize=(12,9), sharey='row')
for i, samples in enumerate([mcmc_result.samples_dict, zltn_result.samples_dict, multinormal_result.samples_dict, laplace_result.samples_dict]):
	for k, var in enumerate(['AV', 'mu', 'theta']):
		subplots_legend[k][i] = var + " " + labels[i]
		ratios = []
		# print(samples[var].shape, true_values[var].shape)

		for j in range(num_to_plot):
			mask = samples[var][j] > true_values[var][j] # num samples greater than true value
			ratios.append(np.sum(mask)/num_samples_per_sn) # divide by 1000 samples per SN
		
		p = stats.ks_2samp(ratios, 1 - np.array(ratios)).pvalue
		# print(var, p)
		# p = stats.skewtest(ratios).pvalue
		ax[k][i].hist(ratios, bins=20, weights = 1/num_to_plot * np.ones(num_to_plot))

		ax[k][i].axvline(np.mean(ratios), linestyle ='dashed')
		ax[k][i].axvline(0.5, linestyle ='solid', color='k', lw=0.6)
		# ax[k][i].set_ylim(0, 0.35)
		
		if p > 1e-6:
			p_coords = (0.055, 0.9)
			ax[k][i].annotate("$p_{KS} = $" + '{:.3f}'.format(round(p, 3)),p_coords , xycoords='axes fraction', fontsize=12)
		else:
			p_coords = (0.02, 0.9)
			ax[k][i].annotate("$p_{KS} = $" + '{:.1e}'.format(Decimal(p)), p_coords, xycoords='axes fraction', fontsize=12)

		ax[0][i].set_title(labels[i], fontsize=18)
		ax[k][i].set_xlabel(prob_labels[var], fontsize=18)
		ax[k][0].set_ylabel("Density", fontsize=18)
		ax[k][i].set_xlim(-0.02,1.02)
		ax[k][i].tick_params(axis='both', labelsize=14)


print(subplots_legend)

plt.tight_layout()
plt.show()

fig.savefig("figures/vsbc.pdf", bbox_inches='tight')

# for i, samples in enumerate([mcmc_result.samples_dict, laplace_result.samples_dict, zltn_result.samples_dict, multinormal_result.samples_dict]):
# 	# for var in ['AV', 'mu', 'theta']:
# 	fig, ax = plt.subplots(3, figsize=(5,12))
# 	for k, var in enumerate(['AV', 'mu', 'theta']):
# 		ratios = []
# 		for j in range(num_to_plot):
# 			mask = samples[var][j]> true_values[k][j]
# 			ratios.append(np.sum(mask)/1000)
			
# 		ax[k].hist(ratios, bins=20)
# 		ax[k].axvline(np.mean(ratios), linestyle ='dashed')
# 		ax[0].set_title(labels[i])
# 		ax[k].set_xlabel(latex_version[var])

# 	plt.show()


# zltn vs true subplots
alpha = 0.5
mcmc_color = 'r'
vi_color = 'b'
axis_fontsize = 12

true_values = {'mu': true_mus, 'theta': true_thetas, 'AV': true_avs}
axlims = {'mu': (-0.075, 0.08), 'theta': (-0.2, 0.02), 'AV': (-0.02, 0.035)}

def plot_compare_to_mcmc_and_truth(point_estimates, uncertainties, title, filename=None):
	fig, ax = plt.subplots(3,3, figsize = (9,9), 
		gridspec_kw = {'hspace':0, 'wspace':0.33}, sharex='col')
	print(title)

	for i, var in enumerate(['mu', 'theta', 'AV']):
		print(var)
		ax[0][i].plot(true_values[var], mcmc_result.point_estimates[var], 'o', fillstyle='none', c=mcmc_color, alpha=alpha, label='MCMC')
		ax[0][i].plot(true_values[var], point_estimates[var], '.', alpha=alpha, c=vi_color, label=title)
		ax[0][i].errorbar(true_values[var], mcmc_result.point_estimates[var], yerr = mcmc_result.stds[var], c=mcmc_color, alpha=alpha, linestyle='None')
		ax[0][i].errorbar(true_values[var], point_estimates[var], yerr = uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
		linspace_vals = np.linspace(min(true_values[var]), max(true_values[var]))
		ax[0][i].plot(linspace_vals, linspace_vals, c='k')
		if i == 0:
			ax[0][i].set_ylabel('Fit value', fontsize = axis_fontsize)
		ax[0][i].set_title(latex_version[var])
		ax[0][i].legend()
		if var=='AV':
			ax[0][i].set_xscale('log')
		# 	ax[1][i].set_xscale('log')
		# 	ax[2][i].set_xscale('log')
		# 	ax[0][i].set_yscale('log')


		standardized_residuals_mcmc = (mcmc_result.point_estimates[var] - true_values[var]) / mcmc_result.stds[var]
		standardized_residuals_vi = (point_estimates[var] - true_values[var])/uncertainties[var]

		print('VI', np.median(standardized_residuals_vi), "MCMC", np.median(standardized_residuals_mcmc))

		ax[1][i].plot(true_values[var], mcmc_result.point_estimates[var] - true_values[var], 'o', fillstyle='none', c=mcmc_color, alpha=alpha, label='MCMC')
		ax[1][i].plot(true_values[var], point_estimates[var] - true_values[var], '.', alpha=alpha, c = vi_color, label=title)
		ax[1][i].errorbar(true_values[var], mcmc_result.point_estimates[var] - true_values[var], yerr = mcmc_result.stds[var], c=mcmc_color, alpha=alpha, linestyle='None')

		ax[1][i].errorbar(true_values[var], point_estimates[var] - true_values[var], yerr = uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
		ax[1][i].axhline(0, color = 'k', lw=0.8)
		if i == 0:
			ax[1][i].set_ylabel('Residual (fit - true)', fontsize = axis_fontsize)
		# ax[1][i].legend()
		# if var=='AV':
		# 	ax[1][i].set_xscale('log')

		# plot a few, evenly spaced representative error bars
		num_vals = 10
		if var=='AV':
			vals = np.logspace(np.log10(min(true_values[var])), np.log10(max(true_values[var])), num_vals)
			idx = np.array([np.argsort(np.abs(true_values[var] - q))[0] for q in vals])
			idx[5] = np.argsort(np.abs(true_values[var] - vals[5]))[1] # to avoid a weirdly small error bar
		else:
			vals = np.linspace(min(true_values[var]), max(true_values[var]), num_vals)
			idx = np.array([np.argsort(np.abs(true_values[var] - q))[0] for q in vals])

		ax[2][i].errorbar(true_values[var][idx], (point_estimates[var] - mcmc_result.point_estimates[var])[idx], 
			yerr = mcmc_result.stds[var][idx], color='r', alpha=0.4, fmt='None', lw=2)

		print("VI - MCMC", np.median((point_estimates[var] - mcmc_result.point_estimates[var])/mcmc_result.stds[var]))

		ax[2][i].plot(true_values[var], point_estimates[var] - mcmc_result.point_estimates[var], 'o', color='gray', alpha = 0.5)
		ax[2][i].axhline(np.median(point_estimates[var] - mcmc_result.point_estimates[var]), color='k', linestyle='dashed', lw=0.8)
		
		# ax[2][i].plot(true_values[var], (point_estimates[var] - mcmc_result.point_estimates[var])/mcmc_result.stds[var], 'o', color='gray', alpha = alpha)


		# ax[2][i].errorbar(true_values[var], point_estimates[var] - mcmc_result.point_estimates[var], yerr = np.max([zltn_result.stds[var], mcmc_result.stds[var]]), c='gray', linestyle='None')
		# ax[2][i].set_ylim(axlims[var])
		ax[2][i].axhline(0, color = 'k', lw=0.8)
		if i==0:
			ax[2][i].set_ylabel('Residual (' + title + ' - MCMC)', fontsize = axis_fontsize)

		# ax[2][i].set_ylabel('Res. ' + latex_version[var] + ' (' + title + ' - MCMC) / $\\sigma$(MCMC)', fontsize = axis_fontsize)
		ax[2][i].set_xlabel('True ' + latex_version[var], fontsize = axis_fontsize)
		# if var=='AV':
		# 	ax[2][i].set_xscale('log')

		# ax[2][2].set_yticks([0.01, 0, -0.01])
		# ax[2][2].set_xticks([0, 0.5,  1, 1.5])
	# ax[0][0].annotate("$\\mu$: median", (36, 34.5), fontsize = 16)
	# ax[0][1].annotate("$\\theta$: median", (0,-1.5), fontsize = 16)

	# if av_stat == 'median':
	# 	ax[0][2].annotate("$A_V$: median", (0.5,0.2), fontsize = 16)
	# elif av_stat == 'mode':
	# 	ax[0][2].annotate("$A_V$: mode", (0.7,0.2), fontsize = 16)


	for axis in ax.flatten():
	  axis.tick_params(axis='x', labelsize=12)
	  axis.tick_params(axis='y', labelsize=12)

	# fig.annotate("$N_{SN} = 1000$", (0.85, 0.97), xycoords='axes fraction')
	ax[0][0].text(0.45,0.05, "$N_{SN} = 1000$", transform=ax[0][0].transAxes, 
            size=12)
	plt.tight_layout()
	plt.show()

	if filename is not None:
		fig.savefig(filename, bbox_inches = 'tight')

	# var = 'AV'
	# plt.plot(true_values[var], (point_estimates[var] - mcmc_result.point_estimates[var]), 'o')
	# plt.fill_between(true_values[var],0, mcmc_result.stds[var], alpha=0.2)
	# plt.xlabel("True AV", fontsize=16)
	# plt.ylabel("$\\frac{ZLTN-MCMC}{\sigma(MCMC)}$", fontsize=16)
	# plt.title(title)
	# plt.show()


	# plt.plot(point_estimates['mu'] - mcmc_result.point_estimates['mu'], point_estimates['theta'] - mcmc_result.point_estimates['theta'], 'o')
	# plt.show()


def plot_marginal_zscores(point_estimates, point_estimates2, title1, title2, filename=None):
	# fig, ax = plt.subplots(3,3, figsize = (9,9), 
	# 	gridspec_kw = {'hspace':0, 'wspace':0.6}, sharex='col')
	fig = plt.figure()
	# colors=['b', 'r', 'g']
	colors=['teal', 'peru', 'mediumorchid']

	bins = np.linspace(-0.15, 0.5, 50)
	for i, var in enumerate(['mu', 'theta', 'AV']):
		num_samples = len(point_estimates[var])

		z_scores = (point_estimates[var] - mcmc_result.point_estimates[var])/mcmc_result.stds[var]
		plt.hist(z_scores, label=latex_version[var], histtype='step', color=colors[i], bins=bins, lw=2)
		if point_estimates2 is not None:
			z_scores2 = (point_estimates2[var] - mcmc_result.point_estimates[var])/mcmc_result.stds[var]
			plt.hist(z_scores2, label=title2+ " " + latex_version[var], histtype='step', color=colors[i], linestyle='dashed', bins=bins)

	plt.legend()
	plt.ylabel("Frequency", fontsize=16)
	plt.xlabel("$\\frac{MVZLTN-MCMC}{\sigma(MCMC)}$", fontsize=16)
	plt.tight_layout()
	plt.show()

	if filename is not None:
		fig.savefig(filename, bbox_inches = 'tight')

#plot_marginal_zscores(zltn_result.point_estimates,None, title1 = "ZLTN", title2="MN", filename='figures/zltn_zscores.pdf')

plot_compare_to_mcmc_and_truth(zltn_result.point_estimates, zltn_result.stds, title = "MVZLTN", filename='figures/zltn_mcmc_sims.pdf')

plot_compare_to_mcmc_and_truth(multinormal_result.point_estimates, multinormal_result.stds, title = "MN", filename='figures/multiinormal_mcmc_sims.pdf')


print(x)


plot_compare_to_mcmc_and_truth(laplace_result.point_estimates, laplace_result.stds, title = "Laplace", filename='figures/laplace_mcmc_sims.pdf')



	# ax[0][0].annotate("$\\mu$: median", (36, 34.5), fontsize = 16)
	# ax[0][1].annotate("$\\theta$: median", (0,-1.5), fontsize = 16)

labels = ['MVZLTN', 'Laplace', 'MultiNormal', 'MCMC']
colors = ['tab:blue', 'tab:red', 'tab:green', 'goldenrod']

fig, ax = plt.subplots(3,2, figsize = (10, 8))
for i, var in enumerate(['mu', 'theta', 'AV']):
	for j, point_estimates in enumerate([zltn_result.point_estimates, laplace_result.point_estimates, multinormal_result.point_estimates, mcmc_result.point_estimates]):
		residuals = point_estimates[var] - true_values[var]
		if i == 0:
			_, bins, _ = ax[i][0].hist(residuals, histtype='step', label=labels[j], color=colors[j])
		else:
			ax[i][0].hist(residuals, histtype='step', label=labels[j], color=colors[j], bins=bins)
		ax[i][0].axvline(np.median(residuals), color=colors[j], linestyle='dashed')
		ax[i][0].set_xlabel(latex_version[var] + " Residual (Fit - True)")
		ax[i][0].legend()
# plt.tight_layout()
# plt.show()

for i, var in enumerate(['mu', 'theta', 'AV']):
	for j, uncertainties in enumerate([zltn_result.stds, laplace_result.stds, multinormal_result.stds, mcmc_result.stds]):
		# residuals = point_estimates[var] - true_values[var]
		ax[i][1].hist(uncertainties[var], histtype='step', label=labels[j], color=colors[j])
		ax[i][1].axvline(np.median(uncertainties[var]), color=colors[j], linestyle='dashed')
		ax[i][1].set_xlabel(latex_version[var] + " Uncertainty (stdev)")
		ax[i][1].legend()
plt.tight_layout()
plt.show()


def uncertainties_corner_plot(point_estimates, title):
	arr = []
	for var in ['AV', 'mu', 'theta']:
		arr.append(point_estimates[var] - true_values[var])

	print(np.array(arr).T.shape)

	fig = corner.corner(np.array(arr).T, plot_contours = False, plot_datapoints =  True, plot_density = False,
              data_kwargs = {'color':'tab:blue', 'alpha':1, 'marker':'o'}, labels = ['$A_V$ residual', '$\\mu$ residual', '$\\theta$ residual'],
              label_kwargs = {'fontsize':12})
	axes = np.array(fig.axes).reshape((3, 3))
	for a in axes[np.triu_indices(3)]:
	  a.remove()
	plt.title(title)
	plt.show()

# uncertainties_corner_plot(zltn_result.point_estimates, "ZLTN")
# uncertainties_corner_plot(laplace_result.point_estimates, "Laplace")
# uncertainties_corner_plot(multinormal_result.point_estimates, "MultiNormal")