import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
import numpy as np
import corner
import pandas as pd

dataset = 'T21_training_set'

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])
sn_names = (sn_list.sn.values)

low = False
if low:
	sn_number = 14
else:
	sn_number = 36
print(sn_names[sn_number])

d2=np.load("foundation_vmap_112023.npy", allow_pickle=True).item()

vi_objects = {}

for var in ['AV', 'mu', 'theta']:
	if var == 'mu':
		vi_objects[var] = np.squeeze(d2['Ds'])[sn_number]
	else:
		vi_objects[var] = np.squeeze(d2[var])[sn_number]


with (open("results/" + dataset + "/" + str(sn_number) + "_mcmc/chains.pkl", "rb")) as openfile:
	mcmc_objects = pickle.load(openfile)


s = np.load("../dist_chains_210610_135216/" + sn_names[sn_number] + "_chains_210610_135216.npy", allow_pickle=True).item()
stephen_mu = s['mu']
stephen_AV = s['AV']
stephen_theta = s['theta']

print(np.mean(stephen_AV))


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
		vi_samples = np.squeeze(vi_objects[var])

		if var == 'AV':
			vi_samples = np.squeeze(vi_objects[var])
		mcmc_results.append(mcmc_samples)
		vi_results.append(vi_samples)

vi_results = np.array(vi_results).T
mcmc_results = np.array(mcmc_results).T
print(vi_results.shape)
print(mcmc_results.shape)

range_low = [(-0.01,0.15), (35, 35.6), (0,1.8)]
range_high = [(0.4, 1), (35, 35.8), (-1.6,-0.2)]

fig = corner.corner(vi_results, labels = ["$A_V$", "$\mu$", "$\\theta$"], 
	range=range_low if low else range_high, label_kwargs = {'fontsize':16})
corner.corner(mcmc_results, color = 'r', fig = fig, range=range_low if low else range_high)

# corner.overplot_lines(fig, [vi_mu[0],vi_mu[-1],vi_mu[1]], linestyle = 'dashed', color='g')
# corner.overplot_lines(fig, [true_av, true_mu, true_theta], linestyle = 'solid', color='blue')

colors = ['k','r']

labels = ['ZLTN VI', 'MCMC']

plt.legend(
    handles=[
        mlines.Line2D([], [], color=colors[i], label=labels[i])
        for i in range(len(labels))
    ],
    fontsize=16, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
)

plt.show()
fig_name = "foundation_single_low.pdf" if low else "foundation_single_high.pdf"

fig.savefig("figures/" + fig_name, bbox_inches='tight')

# Plot comparing to Stephen's MCMC chains
fig = corner.corner(vi_results, labels = ["$A_V$", "$\mu$", "$\\theta$"], 
	range=range_low if low else range_high, label_kwargs = {'fontsize':16})
corner.corner(stephen_data, color = 'r', fig = fig, range=range_low if low else range_high)

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