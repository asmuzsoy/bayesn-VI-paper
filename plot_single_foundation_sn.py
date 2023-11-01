import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
import numpy as np
import corner
import pandas as pd

dataset = 'T21_training_set'

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])
sn_names = (sn_list.sn.values)

sn_number = 14

with (open("results/" + dataset + "/" + str(sn_number) + "_vi/chains.pkl", "rb")) as openfile:
	vi_objects = pickle.load(openfile)

with (open("results/" + dataset + "/" + str(sn_number) + "_mcmc/chains.pkl", "rb")) as openfile:
	mcmc_objects = pickle.load(openfile)


s = np.load("../dist_chains_210610_135216/" + sn_names[sn_number] + "_chains_210610_135216.npy", allow_pickle=True).item()
stephen_mu = s['mu']
stephen_AV = s['AV']
stephen_theta = s['theta']
print(stephen_theta)


stephen_data  = np.array([stephen_AV, stephen_mu, stephen_theta]).T
print(stephen_data.shape)

def get_mode_from_samples(samples):
	hist, bin_edges = np.histogram(samples, bins=50)
	max_index = np.argmax(hist)
	mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
	return mode

print(mcmc_objects['AV'][:,:,0].shape)
# print(objects['AV'].shape)


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

num_sigma = [3,5,3]
# range1 = [(0.5,0.7), (34.3, 35.3), (-1.5,1.5)]

# fig = corner.corner(vi_results, labels = ["$A_V$", "$\mu$", "$\\theta$"], range=range1)
# corner.corner(mcmc_results, color = 'r', fig = fig, range=range1)

fig = corner.corner(vi_results, labels = ["$A_V$", "$\mu$", "$\\theta$"])
corner.corner(mcmc_results, color = 'r', fig = fig)

# corner.overplot_lines(fig, [vi_mu[0],vi_mu[-1],vi_mu[1]], linestyle = 'dashed', color='g')
# corner.overplot_lines(fig, [true_av, true_mu, true_theta], linestyle = 'solid', color='blue')

colors = ['k','r']

labels = ['VI Samples', 'MCMC Samples']

plt.legend(
    handles=[
        mlines.Line2D([], [], color=colors[i], label=labels[i])
        for i in range(len(labels))
    ],
    fontsize=14, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
)

plt.show()

# range1 = [(0.5,0.7), (34.5, 35.5), (-2,0)]


# fig = corner.corner(vi_results, labels = ["$A_V$", "$\mu$", "$\\theta$"], range=range1)
# corner.corner(stephen_data, color = 'r', fig = fig, range=range1)

fig = corner.corner(vi_results, labels = ["$A_V$", "$\mu$", "$\\theta$"])
corner.corner(stephen_data, color = 'r', fig = fig)


# corner.overplot_lines(fig, [vi_mu[0],vi_mu[-1],vi_mu[1]], linestyle = 'dashed', color='g')
# corner.overplot_lines(fig, [true_av, true_mu, true_theta], linestyle = 'solid', color='blue')

colors = ['k','r']

labels = ['VI Samples', 'Stephen MCMC Chains']

plt.legend(
    handles=[
        mlines.Line2D([], [], color=colors[i], label=labels[i])
        for i in range(len(labels))
    ],
    fontsize=14, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
)

plt.show()