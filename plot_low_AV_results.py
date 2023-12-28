import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner
from scipy import special


mcmc_point_estimates = {'AV':[], 'mu':[], 'theta':[]}
zltn_point_estimates = {'AV':[], 'mu':[], 'theta':[]}
laplace_point_estimates = {'AV':[], 'mu':[], 'theta':[]}
normal_point_estimates = {'AV':[], 'mu':[], 'theta':[]}


mcmc_uncertainties = {'AV':[], 'mu':[], 'theta':[]}
zltn_uncertainties = {'AV':[], 'mu':[], 'theta':[]}
laplace_uncertainties = {'AV':[], 'mu':[], 'theta':[]}
normal_uncertainties = {'AV':[], 'mu':[], 'theta':[]}

mcmc_samples = {'AV':[], 'mu':[], 'theta':[]}
zltn_samples = {'AV':[], 'mu':[], 'theta':[]}
laplace_samples = {'AV':[], 'mu':[], 'theta':[]}
normal_samples = {'AV':[], 'mu':[], 'theta':[]}



fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

num_to_plot = 500

dataset_number=27


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

mcmc_file=np.load("low_av_mcmc_120723.npy", allow_pickle=True).item()
laplace_file=np.load("low_av_laplace_120723.npy", allow_pickle=True).item()
zltn_file=np.load("low_av_zltn_120723.npy", allow_pickle=True).item()
multinormal_file=np.load("low_av_multinormal_120723.npy", allow_pickle=True).item()

for file, samples in [(mcmc_file, mcmc_samples), (laplace_file,
	laplace_samples), (zltn_file, zltn_samples), (multinormal_file,
	normal_samples)]:

	if file is mcmc_file:
		for var in ['AV', 'mu', 'theta']:
			if var == 'mu':
				samples[var] = np.squeeze(file['Ds']).reshape(num_to_plot, 1000)
			else:
				samples[var] = np.squeeze(file[var]).reshape(num_to_plot, 1000)

	else:
		for var in ['AV', 'mu', 'theta']:
			if var == 'mu':
				samples[var] = np.squeeze(file['Ds'])
			else:
				samples[var] = np.squeeze(file[var])

labels = ['MCMC', 'Laplace', 'ZLTN', 'MultiNormal']
latex_version = {'mu': '$\\mu$', 'theta': '$\\theta$', 'AV': '$A_V$'}

true_values = [true_avs, true_mus, true_thetas]
for i, samples in enumerate([mcmc_samples, laplace_samples, zltn_samples, normal_samples]):
	# for var in ['AV', 'mu', 'theta']:
	fig, ax = plt.subplots(3, figsize=(5,12))
	for k, var in enumerate(['AV', 'mu', 'theta']):
		ratios = []
		for j in range(num_to_plot):
			mask = samples[var][j]> true_values[k][j]
			ratios.append(np.sum(mask)/1000)
			
		ax[k].hist(ratios, bins=20)
		ax[k].axvline(np.mean(ratios), linestyle ='dashed')
		ax[0].set_title(labels[i])
		ax[k].set_xlabel(latex_version[var])

	plt.show()


for file, point_estimates, uncertainties in [(mcmc_file, mcmc_point_estimates, mcmc_uncertainties), (laplace_file,
	laplace_point_estimates,laplace_uncertainties), (zltn_file, zltn_point_estimates, zltn_uncertainties), (multinormal_file,
	normal_point_estimates, normal_uncertainties)]:

	if file is mcmc_file:
		for var in ['AV', 'mu', 'theta']:
			if var == 'mu':
				point_estimates[var] = np.median(np.squeeze(file['Ds']).reshape(num_to_plot, 1000), axis = 1)
				uncertainties[var] = np.std(np.squeeze(file['Ds']).reshape(num_to_plot, 1000), axis = 1)
			else:
				point_estimates[var] = np.median(np.squeeze(file[var]).reshape(num_to_plot, 1000), axis = 1)
				uncertainties[var] = np.std(np.squeeze(file[var]).reshape(num_to_plot, 1000), axis = 1)

	else:
		for var in ['AV', 'mu', 'theta']:
			if var == 'mu':
				point_estimates[var] = np.median(np.squeeze(file['Ds']), axis = 1)
				uncertainties[var] = np.std(np.squeeze(file['Ds']), axis = 1)
			# elif var == 'AV':
			# 	point_estimates[var] = np.array([get_mode_from_samples(i) for i in np.squeeze(file[var])])
			# 	uncertainties[var] = np.std(np.squeeze(file[var]), axis = 1)
			else:
				point_estimates[var] = np.median(np.squeeze(file[var]), axis = 1)
				uncertainties[var] = np.std(np.squeeze(file[var]), axis = 1)

# Calculate KL divergence with MCMC distribution
# this is wrong becuase we need the probabilities not the samples
# fig, ax = plt.subplots()
# labels = ['ZLTN', 'Laplace', 'MultiNormal']
# colors = ['r', 'b', 'g']
# for j, samples in enumerate([zltn_samples, laplace_samples, normal_samples]):
# 	for i, var in enumerate(['AV']):
# 		print(var)
# 		kldivs = np.sum(special.rel_entr(samples[var], mcmc_samples[var]), axis=1)
# 		print(kldivs)
# 		ax.hist(kldivs, label = labels[j], histtype='step', color = colors[j])
# 		ax.axvline(np.median(kldivs), linestyle='dotted', color=colors[j])
# plt.legend()
# plt.show()
# print(np.sum(special.rel_entr(zltn_samples['AV'], mcmc_samples['AV']), axis=1).shape)


print(mcmc_uncertainties['mu'])
print(zltn_uncertainties['mu'])

# for var in ['AV', 'mu', 'theta']:
# 	mcmc_point_estimates[var] = np.array(mcmc_point_estimates[var])
# 	zltn_point_estimates[var] = np.array(zltn_point_estimates[var])
# 	laplace_point_estimates[var] = np.array(laplace_point_estimates[var])
# 	normal_point_estimates[var] = np.array(normal_point_estimates[var])


# 	zltn_uncertainties[var] = np.array(zltn_uncertainties[var])
# 	mcmc_uncertainties[var] = np.array(mcmc_uncertainties[var])
# 	laplace_uncertainties[var] = np.array(laplace_uncertainties[var])
# 	normal_uncertainties[var] = np.array(normal_uncertainties[var])

# zltn vs true subplots
alpha = 0.2
mcmc_color = 'r'
vi_color = 'b'
axis_fontsize = 12



true_values = {'mu': true_mus, 'theta': true_thetas, 'AV': true_avs}
axlims = {'mu': (-0.075, 0.08), 'theta': (-0.2, 0.02), 'AV': (-0.02, 0.035)}



def plot_compare_to_mcmc_and_truth(point_estimates, uncertainties, title):
	fig, ax = plt.subplots(3,3, figsize = (9,9))

	for i, var in enumerate(['mu', 'theta', 'AV']):
		print(var)
		ax[0][i].plot(true_values[var], point_estimates[var], '.', alpha=alpha, c=vi_color, label=title)
		ax[0][i].plot(true_values[var], mcmc_point_estimates[var], '.', c=mcmc_color, alpha=alpha, label='MCMC')
		ax[0][i].errorbar(true_values[var], point_estimates[var], yerr = uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
		ax[0][i].errorbar(true_values[var], mcmc_point_estimates[var], yerr = mcmc_uncertainties[var], c=mcmc_color, alpha=alpha, linestyle='None')
		linspace_vals = np.linspace(min(true_values[var]), max(true_values[var]))
		ax[0][i].plot(linspace_vals, linspace_vals, c='k')
		ax[0][i].set_ylabel(title + ' '+ latex_version[var], fontsize = axis_fontsize)
		ax[0][i].legend()
		if var=='AV':
			ax[0][i].set_xscale('log')

		ax[1][i].plot(true_values[var], point_estimates[var] - true_values[var], '.', alpha=alpha, c = vi_color, label=title)
		ax[1][i].plot(true_values[var], mcmc_point_estimates[var] - true_values[var], '.', c=mcmc_color, alpha=alpha, label='MCMC')
		ax[1][i].errorbar(true_values[var], point_estimates[var] - true_values[var], yerr = uncertainties[var], alpha=alpha, c=vi_color, linestyle='None')
		ax[1][i].errorbar(true_values[var], mcmc_point_estimates[var] - true_values[var], yerr = mcmc_uncertainties[var], c=mcmc_color, alpha=alpha, linestyle='None')
		ax[1][i].axhline(0, color = 'k')
		ax[1][i].set_ylabel('Residual ' + latex_version[var] + ' (' + title + ' - true)', fontsize = axis_fontsize)
		ax[1][i].legend()
		if var=='AV':
			ax[1][i].set_xscale('log')

		ax[2][i].plot(true_values[var], point_estimates[var] - mcmc_point_estimates[var], 'o', color='gray', alpha = alpha)
		# ax[2][i].errorbar(true_values[var], point_estimates[var] - mcmc_point_estimates[var], yerr = np.max([zltn_uncertainties[var], mcmc_uncertainties[var]]), c='gray', linestyle='None')
		# ax[2][i].set_ylim(axlims[var])
		ax[2][i].axhline(0, color = 'k')
		ax[2][i].set_ylabel('Residual ' + latex_version[var] + ' (' + title + ' - MCMC)', fontsize = axis_fontsize)
		ax[2][i].set_xlabel('True ' + latex_version[var], fontsize = axis_fontsize)
		if var=='AV':
			ax[2][i].set_xscale('log')
	# ax[0][0].annotate("$\\mu$: median", (36, 34.5), fontsize = 16)
	# ax[0][1].annotate("$\\theta$: median", (0,-1.5), fontsize = 16)

	# if av_stat == 'median':
	# 	ax[0][2].annotate("$A_V$: median", (0.5,0.2), fontsize = 16)
	# elif av_stat == 'mode':
	# 	ax[0][2].annotate("$A_V$: mode", (0.7,0.2), fontsize = 16)


	for axis in ax.flatten():
	  axis.tick_params(axis='x', labelsize=12)
	  axis.tick_params(axis='y', labelsize=12)

	plt.tight_layout()
	plt.show()


	# plt.plot(point_estimates['mu'] - mcmc_point_estimates['mu'], point_estimates['theta'] - mcmc_point_estimates['theta'], 'o')
	# plt.show()


plot_compare_to_mcmc_and_truth(zltn_point_estimates, zltn_uncertainties, title = "ZLTN")
plot_compare_to_mcmc_and_truth(laplace_point_estimates, laplace_uncertainties, title = "Laplace")
print("normal")
plot_compare_to_mcmc_and_truth(normal_point_estimates, normal_uncertainties, title = "MultiNormal")


	# ax[0][0].annotate("$\\mu$: median", (36, 34.5), fontsize = 16)
	# ax[0][1].annotate("$\\theta$: median", (0,-1.5), fontsize = 16)

labels = ['ZLTN', 'Laplace', 'MultiNormal', 'MCMC']
colors = ['tab:blue', 'tab:red', 'tab:green', 'goldenrod']

fig, ax = plt.subplots(3, figsize = (3, 10))
for i, var in enumerate(['mu', 'theta', 'AV']):
	for j, point_estimates in enumerate([zltn_point_estimates, laplace_point_estimates, normal_point_estimates, mcmc_point_estimates]):
		residuals = point_estimates[var] - true_values[var]
		if i == 0:
			_, bins, _ = ax[i].hist(residuals, histtype='step', label=labels[j], color=colors[j])
		else:
			ax[i].hist(residuals, histtype='step', label=labels[j], color=colors[j], bins=bins)
		ax[i].axvline(np.median(residuals), color=colors[j], linestyle='dashed')
		ax[i].set_xlabel(latex_version[var] + " Residual (Fit - True)")
		ax[i].legend()
plt.show()

fig, ax = plt.subplots(3, figsize = (3, 10))
for i, var in enumerate(['mu', 'theta', 'AV']):
	for j, uncertainties in enumerate([zltn_uncertainties, laplace_uncertainties, normal_uncertainties, mcmc_uncertainties]):
		# residuals = point_estimates[var] - true_values[var]
		ax[i].hist(uncertainties[var], histtype='step', label=labels[j], color=colors[j])
		ax[i].axvline(np.median(uncertainties[var]), color=colors[j], linestyle='dashed')
		ax[i].set_xlabel(latex_version[var] + " Uncertainty (stdev)")
		ax[i].legend()
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

# uncertainties_corner_plot(zltn_point_estimates, "ZLTN")
# uncertainties_corner_plot(laplace_point_estimates, "Laplace")
# uncertainties_corner_plot(normal_point_estimates, "MultiNormal")

