import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pickle
import numpy as np
import corner


# dataset = 'T21_sim_2'
dataset = 'sim_really_low_AV'
# dataset = 'sim_low_AV'
# dataset = 'sim_nonzero_eps'



with (open("results/" + dataset + "_vi/chains.pkl", "rb")) as openfile:
	vi_objects = pickle.load(openfile)

with (open("results/" + dataset + "_mcmc/chains.pkl", "rb")) as openfile:
	mcmc_objects = pickle.load(openfile)


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
		mcmc_results.append(mcmc_samples)
		vi_results.append(vi_samples)

	vi_results = np.array(vi_results).T
	mcmc_results = np.array(mcmc_results).T
	print(vi_results.shape)
	print(mcmc_results.shape)

	range1 = [(-0.05, 0.4), (34, 35), (1.5, 2.5)]


	fig = corner.corner(vi_results, labels = ["AV", "mu", "theta"])
	corner.corner(mcmc_results, color = 'r', fig = fig)
	if dataset == 'sim_low_AV':
		corner.overplot_lines(fig, [known_values['AV'],known_values['mu'],known_values['theta']], linestyle = 'dashed', color='b')
	if dataset == 'sim_really_low_AV':
		corner.overplot_lines(fig, [lower_known_values['AV'],lower_known_values['mu'],lower_known_values['theta']])
	if dataset == 'sim_nonzero_eps':
		corner.overplot_lines(fig, [nonzero_eps_values['AV'],nonzero_eps_values['mu'],nonzero_eps_values['theta']])
	

	colors = ['k','r', 'b']

	labels = ['VI Fit', 'MCMC', 'True Values']

	plt.legend(
	    handles=[
	        mlines.Line2D([], [], color=colors[i], label=labels[i])
	        for i in range(len(labels))
	    ],
	    fontsize=14, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
	)


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
