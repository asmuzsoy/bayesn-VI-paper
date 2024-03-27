import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
import numpy as np
import corner
import pandas as pd
from stephen_corner_plots import *
# import pairplots


dataset = 'T21_training_set'

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])
sn_names = (sn_list.sn.values)

low = True
if low:
	sn_number = 148
else:
	sn_number = 36
print(sn_names[sn_number])

# zltn_dict=np.load("foundation_results/foundation_vmap_zltn_011724_samples.npy", allow_pickle=True).item()
# laplace_dict=np.load("foundation_results/foundation_vmap_laplace_011724_samples.npy", allow_pickle=True).item()
# multinormal_dict=np.load("foundation_results/foundation_vmap_multinormal_011724_samples.npy", allow_pickle=True).item()
# mcmc_dict=np.load("foundation_vmap_mcmc_122923.npy", allow_pickle=True).item()
zltn_dict=np.load("foundation_results/foundation_vmap_zltn_032624_samples.npy", allow_pickle=True).item()
laplace_dict=np.load("foundation_results/foundation_vmap_laplace_032624_samples.npy", allow_pickle=True).item()
multinormal_dict=np.load("foundation_results/foundation_vmap_multinormal_032624_samples.npy", allow_pickle=True).item()
mcmc_dict=np.load("foundation_results/foundation_vmap_mcmc_032624_samples.npy", allow_pickle=True).item()


median_avs = np.median(mcmc_dict['AV'].reshape((157,1000)),axis=1)

s = np.load("../dist_chains_210610_135216/" + sn_names[sn_number] + "_chains_210610_135216.npy", allow_pickle=True).item()
stephen_mu = s['mu']
stephen_AV = s['AV']
stephen_theta = s['theta']

# print(np.mean(stephen_AV))


stephen_data  = np.array([stephen_AV, stephen_mu, stephen_theta]).T
# print(stephen_data.shape)

def get_mode_from_samples(samples):
	hist, bin_edges = np.histogram(samples, bins=50)
	max_index = np.argmax(hist)
	mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
	return mode


mcmc_results = []
zltn_results= []
laplace_results = []
multinormal_results = []

for var in ['AV', 'mu', 'theta']:
	mcmc_samples = np.squeeze(mcmc_dict[var])[sn_number].reshape((1000,))
	# print(var, np.squeeze(mcmc_dict[var]).shape)
	zltn_samples = np.squeeze(zltn_dict[var])[sn_number]
	laplace_samples = np.squeeze(laplace_dict[var])[sn_number]
	multinormal_samples = np.squeeze(multinormal_dict[var])[sn_number]
	# print(laplace_samples)
	mcmc_results.append(mcmc_samples)
	zltn_results.append(zltn_samples)
	laplace_results.append(laplace_samples)
	multinormal_results.append(multinormal_samples)

zltn_results = np.array(zltn_results).T
mcmc_results = np.array(mcmc_results).T
laplace_results = np.array(laplace_results).T
multinormal_results = np.array(multinormal_results).T

# range_low = [[-0.01,0.2], [36.5, 37.5], [0.7,2.5]]
# bounds=[[-0.1,None], [None, None], [None, None]]
# range_high = [(0.4, 1), (35, 35.8), (-1.6,-0.2)]
# smoothing = 1.1

# fig, ax = stephen_corner(zltn_results.T, 
# 			names=["$A_V$", "$\\mu$", "$\\theta$"], 
# 			colour = 'k', 
# 			lims=range_low if low else range_high, 
# 			bounds=bounds, smoothing=smoothing)
# fig, ax = stephen_corner(mcmc_results.T, 
# 			names=["$A_V$", "$\\mu$", "$\\theta$"], 
# 				fig_ax = (fig, ax), colour = 'red', 
# 				lims=range_low if low else range_high, 
# 				bounds=bounds, smoothing=smoothing)
# fig, ax = stephen_corner(laplace_results.T, 
# 			names=["$A_V$", "$\\mu$", "$\\theta$"], 
# 			fig_ax = (fig, ax), colour = 'g', 
# 			lims=range_low if low else range_high, 
# 			smoothing=smoothing)
# fig, ax = stephen_corner(multinormal_results.T, 
# 			names=["$A_V$", "$\\mu$", "$\\theta$"], 
# 				fig_ax = (fig, ax), colour = 'b', 
# 				lims=range_low if low else range_high, 
# 				smoothing=smoothing)
# colors = ['r', 'k', 'b', 'g']

# labels = [ 'MCMC', 'ZLTN VI','Multivariate Normal VI', 'Laplace Approximation']

# plt.legend(
#     handles=[
#         mlines.Line2D([], [], color=colors[i], label=labels[i])
#         for i in range(len(labels))
#     ],
#     fontsize=16, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
# )

# plt.show()

# args = (pairplots.Contour(),pairplots.MarginDensity())

# pairplots.pairplot_interactive(zltn_results, mcmc_results, 
# 	laplace_results, multinormal_results,labels = {
#     '1':pairplots.latex(r"A_V"),
#     # Makie rich text
#     '2':pairplots.latex(r"\mu"),
#     # LaTeX String
#     '3':pairplots.latex(r"\theta"),
# })
# pairplots.pairplot_interactive((pairplots.series(zltn_results, label='ZLTN VI'), args),
# 	(pairplots.series(mcmc_results, label='MCMC'),args), (pairplots.series(laplace_results, label='Laplace Approximation'),args),
# 	(pairplots.series(multinormal_results, label='Multivariate Normal VI'), args), labels=["$A_V$", "$\mu$", "$\\theta$"])
range_low = [(-0.01,0.2), (35, 35.6), (-1.8,1.8)]

range_low = [(-0.01,0.2), (36.5, 37.5), (0.7,2.5)]
range_high = [(0.4, 1), (35, 35.8), (-1.6,-0.2)]
factor = 0.5
fig = corner.corner(zltn_results, 
	labels = ["$A_V$", "$\\mu$", "$\\theta$"], 
	range=range_low if low else range_high, 
	label_kwargs = {'fontsize':16})
corner.corner(mcmc_results,  color = 'r', fig = fig, 
	range=range_low if low else range_high)
corner.corner(laplace_results, color = 'g', fig = fig, 
	range=range_low if low else range_high)
corner.corner(multinormal_results, color = 'b', fig = fig, 
	range=range_low if low else range_high)


colors = ['r', 'k', 'b', 'g']

labels = [ 'MCMC', 'ZLTN VI','Multivariate Normal VI', 'Laplace Approximation']

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
fig = corner.corner(zltn_results, labels = ["$A_V$", "$\mu$", "$\\theta$"], 
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