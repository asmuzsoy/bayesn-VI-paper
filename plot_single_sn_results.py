import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pickle
import numpy as np
import corner
from astropy.cosmology import FlatLambdaCDM


fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)


# dataset = 'sim_population_15/2'
# dataset = 'sim_really_low_AV'
# dataset = 'sim_low_AV'
# dataset = 'sim_zero_AV'

# dataset = 'sim_population_15/1'
dataset_number = 15
sn_number = 1
dataset = 'sim_population_' + str(dataset_number) + '/' + str(sn_number)
true_av = np.loadtxt("sim_population_AV_" + str(dataset_number) + ".txt")[sn_number]
true_theta = np.loadtxt("sim_population_theta_" + str(dataset_number) + ".txt")[sn_number]
true_z = np.loadtxt("sim_population_z_" + str(dataset_number) + ".txt")[sn_number]
true_mu = cosmo.distmod(true_z).value


with (open("results/" + dataset + "_vi/chains.pkl", "rb")) as openfile:
	vi_objects = pickle.load(openfile)

with (open("results/" + dataset + "_mcmc/chains.pkl", "rb")) as openfile:
	mcmc_objects = pickle.load(openfile)


vi_params = np.load("results/" + dataset + "_vi/vi_params.npz")
vi_mu = vi_params['mu']
vi_cov = vi_params['cov']

print(vi_mu)

print(vi_mu)
print(vi_objects.keys())
print(vi_objects['AV'][:,0].shape)
print(vi_objects['AV'][:,1].shape)
print(vi_objects['AV'].shape)
print(vi_objects['AV'][0][0].shape)

# print(vi_objects['AV'])




print(mcmc_objects['AV'][:,:,0].shape)
# print(objects['AV'].shape)

known_values = {'AV': 0.1, 'mu':34.59932899, 'theta':2.}

lower_known_values = {'AV': 0.01, 'mu':34.59932899, 'theta':2.}

nonzero_eps_values = {'AV': 0.05, 'mu':34.59932899, 'theta':2.}


for i in range(1):
	mcmc_results = []
	vi_results = []
	for var in ['AV', 'mu', 'theta']:
		mcmc_samples = mcmc_objects[var][:,:,i].reshape((1000,))
		vi_samples = np.squeeze(vi_objects[var][:,i])

		if var == 'AV':
			vi_samples = np.squeeze(vi_objects[var][:,i])
		mcmc_results.append(mcmc_samples)
		vi_results.append(vi_samples)

	vi_results = np.array(vi_results).T
	mcmc_results = np.array(mcmc_results).T
	print(vi_results.shape)
	print(mcmc_results.shape)

	range1 = [(-0.05,0.2), (34, 35), (-0.5,0.5)]
	num_sigma = 3
	range1 = [((vi_mu[0] - num_sigma * np.sqrt(vi_cov[0][0]))[0], (vi_mu[0] + num_sigma * np.sqrt(vi_cov[0][0]))[0]), ((vi_mu[-1] - (num_sigma+1) * np.sqrt(vi_cov[-1][-1]))[0], (vi_mu[-1] + (num_sigma+1) * np.sqrt(vi_cov[-1][-1]))[0]), ((vi_mu[1] - num_sigma * np.sqrt(vi_cov[1][1]))[0], (vi_mu[1] + num_sigma * np.sqrt(vi_cov[1][1]))[0])]


	fig = corner.corner(vi_results, labels = ["AV", "mu", "theta"], range=range1)
	corner.corner(mcmc_results, color = 'r', fig = fig, range=range1)
	if dataset == 'sim_low_AV':
		corner.overplot_lines(fig, [known_values['AV'],known_values['mu'],known_values['theta']], linestyle = 'dashed', color='b')
	if dataset == 'sim_really_low_AV':
		corner.overplot_lines(fig, [lower_known_values['AV'],lower_known_values['mu'],lower_known_values['theta']])
	if dataset == 'sim_nonzero_eps':
		corner.overplot_lines(fig, [nonzero_eps_values['AV'],nonzero_eps_values['mu'],nonzero_eps_values['theta']])
	
	corner.overplot_lines(fig, [vi_mu[0],vi_mu[-1],vi_mu[1]], linestyle = 'dashed', color='g')
	corner.overplot_lines(fig, [true_av, true_mu, true_theta], linestyle = 'solid', color='blue')

	colors = ['k','r', 'b', 'g']

	labels = ['VI Samples', 'MCMC Samples', 'True Values', 'VI parameters']

	plt.legend(
	    handles=[
	        mlines.Line2D([], [], color=colors[i], label=labels[i])
	        for i in range(len(labels))
	    ],
	    fontsize=14, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
	)


	plt.show()

	ds_samples = np.squeeze(vi_objects['Ds'][:,i])
	mu_samples = np.squeeze(vi_objects['mu'][:,i])
	mcmc_mu_samples= mcmc_objects['mu'][:,:,i].reshape((1000,))

	plt.hist(ds_samples, histtype='step', density=True, label='Ds VI samples')
	plt.hist(mu_samples, histtype='step', density=True, label='mu vi samples')
	plt.hist(mcmc_mu_samples, histtype='step', density=True, label='mu MCMC samples')
	plt.axvline(vi_mu[-1], color='tab:blue')
	plt.axvline(np.median(mcmc_mu_samples), color='tab:green')
	plt.legend()
	plt.show()

	delM_samples = vi_objects['delM'][:,i]
	plt.hist(np.squeeze(delM_samples), histtype='step')
	plt.axvline(np.median(delM_samples))
	plt.xlabel("delM")
	plt.show()
		# plt.hist(vi_samples, label = 'VI', histtype='step', density=True)
		# plt.hist(mcmc_samples, label = 'MCMC', histtype='step', density=True)
		# if dataset == 'sim_low_AV':
		# 	plt.axvline(known_values[var], color='k', linestyle = 'dashed', label='true value')
		# if dataset == 'sim_really_low_AV':
		# 	plt.axvline(lower_known_values[var], color='k', linestyle = 'dashed', label='true value')
		# plt.xlabel(var)
		# plt.legend()
		# plt.show()
