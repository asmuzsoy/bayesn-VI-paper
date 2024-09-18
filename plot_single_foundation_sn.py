import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
import numpy as np
import corner
import pandas as pd
# from stephen_corner_plots import *
# import pairplots
import seaborn as sns
from scipy import stats, integrate

dataset = 'T21_training_set'

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])
sn_names = (sn_list.sn.values)

low = True
if low:
	sn_number = 148
else:
	sn_number = 36
print(sn_names[sn_number])

zltn_dict=np.load("foundation_results/foundation_vmap_zltn_032624_samples.npy", allow_pickle=True).item()
laplace_dict=np.load("foundation_results/foundation_vmap_laplace_032624_samples.npy", allow_pickle=True).item()
multinormal_dict=np.load("foundation_results/foundation_vmap_multinormal_032624_samples.npy", allow_pickle=True).item()
mcmc_dict=np.load("foundation_results/foundation_vmap_mcmc_032624_samples.npy", allow_pickle=True).item()

# These are the actual variational parameters (mu and cov matrix) saved from the fit
zltn_params = np.load("foundation_results/foundation_vmap_zltn_090924_params.npz")
mn_params = np.load("foundation_results/foundation_vmap_multinormal_090924_params.npz")
laplace_params = np.load("foundation_results/foundation_vmap_laplace_090924_params.npz")



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
# range_low = [(-0.01,0.2), (35, 35.6), (-1.8,1.8)]

range_low = [(-0.01,0.2), (36.5, 37.5), (0.7,2.5)]
range_high = [(0.4, 1), (35, 35.8), (-1.6,-0.2)]
factor = 0.5
fig = corner.corner(zltn_results,
	labels = ["$A_V$", "$\\mu$", "$\\theta_1$"],
	range=range_low if low else range_high,
	label_kwargs = {'fontsize':24})
corner.corner(mcmc_results,  color = 'r', fig = fig,
	range=range_low if low else range_high)
corner.corner(laplace_results, color = 'g', fig = fig,
	range=range_low if low else range_high)
corner.corner(multinormal_results, color = 'b', fig = fig,
	range=range_low if low else range_high)
axes = fig.get_axes()
for ax in axes:
	# print(ax.get_xticks())
	ax.tick_params(axis='both', labelsize=16)

# remove a pesky label that looks bad
if low:
	axes[6].set_xticks(axes[6].get_xticks()[:-1])
	# axes[3].set_yticks(axes[3].get_yticks()[1:])
else:
	axes[3].set_yticks(axes[3].get_yticks()[:-1])
	axes[7].set_xticks(axes[7].get_xticks()[:-1])


colors = ['r', 'k', 'b', 'g']

labels = [ 'MCMC', 'MVZLTN VI','Multivariate Normal VI', 'Laplace Approximation']

plt.legend(
    handles=[
        mlines.Line2D([], [], color=colors[i], label=labels[i])
        for i in range(len(labels))
    ],
    fontsize=18, frameon=False, bbox_to_anchor=(1, 3), loc="upper right"
)
plt.tight_layout()

plt.show()
fig_name = "foundation_single_low.pdf" if low else "foundation_single_high.pdf"

fig.savefig("figures/" + fig_name, bbox_inches='tight')

# # This function is from Stephen
# def hist_contour(x, y, w=None, bins=64, levels=[0.95, 0.68]):
#     c_, x_, y_ = np.histogram2d(x, y, bins=64, weights=w)
#     x__ = 0.5*(x_[1:] + x_[:-1])
#     y__ = 0.5*(y_[1:] + y_[:-1])
#     f_ = np.sort(c_, axis=None)
#     s_ = np.cumsum(f_)
#     s_ = s_/s_[-1]
#     l_ = [f_[np.argmin(np.fabs(s_ - (1-l)))] for l in levels]
#     return x__, y__, c_.T, l_, x_, y_

print(mcmc_results.shape)

# print(this_vi_mu[0], np.sqrt(this_vi_var))
def normal(x, mu, var):
	sigma = np.sqrt(var)
	constant = 1 / np.sqrt(2 * np.pi * sigma**2)
	return constant * np.exp((-(x - mu)**2 )/(2 * sigma**2))

def lognormal(x, mu, var):
	sigma = np.sqrt(var)
	constant = 1 / np.sqrt(2 * np.pi * sigma**2)
	return (constant / x) * np.exp((-(np.log(x) - mu)**2 )/(2 * sigma**2))

def single_zltn(single_x, mu, var):
	sigma = np.sqrt(var)
	if single_x < 0:
		return 0
	return (1 / sigma) * stats.norm.pdf((single_x - mu)/sigma) /(1 - stats.norm.cdf(-mu/sigma))


def zltn(x, mu, var):
	return np.array([single_zltn(i, mu, var) for i in (x)])
	# return stats.truncnorm.pdf(x, (0 - mu)/np.sqrt(var), np.inf, mu, np.sqrt(var))




x = np.linspace(-0.01,0.2, 1000)

plt.hist(mcmc_results[:,0], density=True, histtype='step', color='r', label="MCMC", bins=30)

zltn_mean = zltn_params['mu'][sn_number][0][0]
mn_mean = mn_params['mu'][sn_number][0]
laplace_mean = laplace_params['mu'][sn_number][0]

zltn_variance = zltn_params['cov'][sn_number][0][0]
mn_variance = mn_params['cov'][sn_number][0][0]
laplace_variance = laplace_params['cov'][sn_number][0][0]


plt.plot(x, zltn(x, zltn_mean, zltn_variance), color = "k", label="MVZLTN")
plt.plot(x, lognormal(x, mn_mean, mn_variance), color = "blue", label="Multivariate Normal")
plt.plot(x, lognormal(x, laplace_mean, laplace_variance), color="green", label="Laplace Approximation")
plt.legend()
plt.xlabel("$A_V$")
plt.ylabel("Density")
plt.show()

fig2, ax = plt.subplots(1,2, figsize=(10,5))
heights, bins, _ = ax[0].hist(mcmc_results[:,0], density=True, histtype='step', color='r', label="MCMC", bins=20, lw=1.5)
ax[0].plot(x, zltn(x, zltn_mean, zltn_variance), color = "k", label="MVZLTN")
ax[0].plot(x, lognormal(x, mn_mean, mn_variance), color = "blue", label="Multivariate Normal")
ax[0].plot(x, lognormal(x, laplace_mean, laplace_variance), color="green", label="Laplace Approximation")

x_for_integration = np.linspace(1e-10, 0.5)

## Make sure everything integrates to 1
print("zltn integration:", integrate.trapz(zltn(x_for_integration, zltn_mean, zltn_variance), x=x_for_integration))
print("multinormal integration:", integrate.trapz(lognormal(x_for_integration, mn_mean, mn_variance), x=x_for_integration))
print("laplace integration:", integrate.trapz(lognormal(x_for_integration, laplace_mean, laplace_variance), x=x_for_integration))

bin_width = bins[1] - bins[0]
print("MCMC hist integration:", bin_width * sum(heights))

h1, b1, _ = ax[1].hist(mcmc_results[:,0], density=True, histtype='step', color='r', label="MCMC", bins=20, lw=1.5)
h2, b2, _ = ax[1].hist(zltn_results[:,0], density=True, histtype='step', color='k', label="MVZLTN", bins=20, lw=1.5)
h3, b3, _ = ax[1].hist(multinormal_results[:,0], density=True, histtype='step', color='b', label="Multivariate Normal", bins=20, lw=1.5)
h4, b4, _ = ax[1].hist(laplace_results[:,0], density=True, histtype='step', color='g', label="Laplace Approximation", bins=100, lw=1.5)
ax[1].set_xlim(-0.01, 0.2)
ax[0].legend()
ax[0].set_xlabel("$A_V$")
ax[0].set_ylabel("Density")
ax[1].legend()
ax[1].set_xlabel("$A_V$")
ax[1].set_ylabel("Density")
# plt.plot(x, stats.lognorm.pdf(x, scale=1, loc = np.exp(mn_params['mu'][sn_number][0]), s = np.sqrt(mn_params['cov'][sn_number][0][0])))
plt.show()

for h, b in zip([h1, h2, h3, h4], [b1, b2, b3, b4]):
	bin_width = b[1] - b[0]
	print(bin_width * sum(h))

## Making actual figure
f = plt.figure(figsize=(6, 11))
subfigs = f.subfigures(2, 1, height_ratios=[0.4, 0.6])
ax_top = subfigs[0].subplots(1, 1)
ax_top.hist(mcmc_results[:,0], density=True, histtype='step', color='r', label="MCMC", bins=20, lw=1.5)
ax_top.plot(x, zltn(x, zltn_mean, zltn_variance), color = "k", label="MVZLTN")
ax_top.plot(x, lognormal(x, mn_mean, mn_variance), color = "blue", label="Multivariate Normal")
ax_top.plot(x, lognormal(x, laplace_mean, laplace_variance), color="green", label="Laplace Approximation")
ax_top.legend(fontsize=14)
ax_top.set_xlabel("$A_V$", fontsize=20)
ax_top.set_ylabel("Density", fontsize=20)
ax_top.tick_params(axis='both', labelsize=14)
ax_top.set_xlim(-0.008, 0.2)

corner.corner(zltn_results,
	labels = ["$A_V$", "$\\mu$", "$\\theta_1$"],
	range=range_low if low else range_high,
	label_kwargs = {'fontsize':20}, fig = subfigs[1])
corner.corner(mcmc_results,  color = 'r', fig = subfigs[1],
	range=range_low if low else range_high)
corner.corner(laplace_results, color = 'g', fig = subfigs[1],
	range=range_low if low else range_high)
corner.corner(multinormal_results, color = 'b', fig = subfigs[1],
	range=range_low if low else range_high)
axes = subfigs[1].get_axes()
for ax in axes:
	# print(ax.get_xticks())
	ax.tick_params(axis='both', labelsize=14)

# remove a pesky label that looks bad
if low:
	axes[6].set_xticks(axes[6].get_xticks()[:-1])
	# axes[3].set_yticks(axes[3].get_yticks()[1:])
else:
	axes[3].set_yticks(axes[3].get_yticks()[:-1])
	axes[7].set_xticks(axes[7].get_xticks()[:-1])

fig_name = "foundation_single_low_new.pdf" if low else "foundation_single_high_new.pdf"


colors = ['r', 'k', 'b', 'g']

labels = [ 'MCMC', 'MVZLTN VI','Multivariate Normal VI', 'Laplace Approximation']

plt.legend(
    handles=[
        mlines.Line2D([], [], color=colors[i], label=labels[i])
        for i in range(len(labels))
    ],
    fontsize=14, frameon=False, bbox_to_anchor=(1, 3), loc="upper right"
)
f.savefig("figures/" + fig_name, bbox_inches='tight')

plt.show() 

# x, y, h, l, _, _ = hist_contour(mcmc_results[:,0], mcmc_results[:,1], bins=20, levels=[0.95, 0.68])
# print(x.shape, y.shape, h.shape)
# print(l)
# plt.contour(x, y, h, levels=[0.68, 0.95], colors="b", linewidths=2)
# plt.xlim(-0.1, 0.5)
# plt.show()

# adjust = 0.1
# sns.kdeplot(mcmc_results[:,0], lw=3, label='MCMC', clip=(0,0.4), bw_adjust=adjust)
# sns.kdeplot(zltn_results[:,0], lw=3, label='ZLTN', clip=(0,0.4), bw_adjust=adjust)
# sns.kdeplot(laplace_results[:,0], lw=3, label='Laplace', clip=(0,0.4), bw_adjust=adjust)
# sns.kdeplot(multinormal_results[:,0], lw=3, label='Multinormal',clip=(0,0.4), bw_adjust=adjust)
# # plt.xlim(-0.02,0.2)
# # plt.xlim(-0.02,0.2)

# plt.title("bw_adjust= " + str(adjust))
# plt.xlabel("$A_V$", fontsize=16)
# plt.ylabel("Frequency", fontsize=16)
# plt.legend()
# plt.show()

# Plot comparing to Stephen's MCMC chains
fig = corner.corner(zltn_results, labels = ["$A_V$", "$\\mu$", "$\\theta$"],
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
