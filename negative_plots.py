import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner
from scipy import special, stats

fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

num_to_plot = 500

dataset_number=28

av_stat = 'median'

title_str = ""

true_avs = np.loadtxt("sim_population_AV_" + str(dataset_number) + ".txt")[:num_to_plot]
true_thetas = np.loadtxt("sim_population_theta_" + str(dataset_number) + ".txt")[:num_to_plot]
true_z = np.loadtxt("sim_population_z_" + str(dataset_number) + ".txt")[:num_to_plot]
true_mus = np.array([cosmo.distmod(z).value for z in true_z])

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

mcmc_result = Result("sim28_vmap_mcmc_020524_samples.npy")
mcmc_result_negative = Result("sim28_vmap_mcmc_040424_samples.npy")

negative_mask = mcmc_result_negative.point_estimates['AV'] < 0

# negative_mask = true_avs < 0.15

rng = np.random.default_rng()

num_times_to_boostrap = 100
def bootstrap_dataset(result):
	new_samples = {}
	for key in result.samples_dict.keys():
		new_samples[key] = rng.choice(result.samples_dict[key], size=result.samples_dict[key].shape[1], replace=True, axis=1)
	return new_samples

def get_medians(result, negative_result, true_vals, num_times=num_times_to_boostrap):
	medians = []
	negative_medians = []
	masked_medians = []
	masked_negative_medians = []
	for i in range(num_times):
		new_samples = bootstrap_dataset(result)
		new_negative_samples = bootstrap_dataset(negative_result)
		av_medians = np.median(new_negative_samples['AV'], axis=1)
		negative_mask = av_medians < 0
		# print(sum(negative_mask))
		residuals = np.median(new_samples['mu'], axis=1) - true_vals
		negative_residuals = np.median(new_negative_samples['mu'], axis=1)  - true_vals
		medians.append(np.median(residuals))
		negative_medians.append(np.median(negative_residuals))
		masked_medians.append(np.median(residuals[negative_mask]))
		masked_negative_medians.append(np.median(negative_residuals[negative_mask]))
	return medians, negative_medians, masked_medians, masked_negative_medians
		

medians, negative_medians, masked_medians, masked_negative_medians = get_medians(mcmc_result, mcmc_result_negative, true_mus)

fig, ax = plt.subplots(1,2, figsize=(8,4))
im = ax[0].scatter(true_mus, mcmc_result.point_estimates['mu'] - true_mus,
	c = mcmc_result_negative.point_estimates['AV'], cmap='bwr',norm=TwoSlopeNorm(0))
negative_arr = (mcmc_result.point_estimates['mu'] - true_mus)[negative_mask]
for i in range(num_times_to_boostrap):
	ax[0].axhline(medians[i], color='dimgray', alpha = 0.1)
	ax[0].axhline(masked_medians[i], color='cornflowerblue', alpha = 0.1)
ax[0].axhline(np.median(negative_arr), linestyle='dashed', color='b', label='median residual (Unc. AV < 0)')
ax[0].axhline(np.median(mcmc_result.point_estimates['mu'] - true_mus), linestyle='dashed', color='k', label='median residual')

plt.colorbar(im, label='Unconstrained Fit $A_V$')
ax[0].axhline(0, color='k', linewidth=0.5)
ax[0].set_title("AV Constrained to be positive")
ax[0].set_ylim(-0.5, 0.5)
ax[0].set_xlabel("True $\\mu$")
ax[0].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")
ax[0].legend()


im = ax[1].scatter(true_mus, mcmc_result_negative.point_estimates['mu'] - true_mus, 
	c = mcmc_result_negative.point_estimates['AV'], cmap='bwr',norm=TwoSlopeNorm(0))

negative_arr = (mcmc_result_negative.point_estimates['mu'] - true_mus)[negative_mask]
for i in range(num_times_to_boostrap):
	ax[1].axhline(negative_medians[i], color='dimgray', alpha = 0.1)
	ax[1].axhline(masked_negative_medians[i], color='cornflowerblue', alpha = 0.1)
ax[1].axhline(np.median(negative_arr), linestyle='dashed', color='b', label='median residual (Unc. AV < 0)')
ax[1].axhline(np.median(mcmc_result_negative.point_estimates['mu'] - true_mus), 
	linestyle='dashed', label='median residual', color='k')
plt.colorbar(im, label='Fit $A_V$')
ax[1].set_title("AV not constrained to be positive")
ax[1].axhline(0, color='k', linewidth=0.5)
ax[1].set_ylim(-0.5, 0.5)
ax[1].set_xlabel("True $\\mu$")
ax[1].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")
ax[1].legend()
plt.show()

# medians, negative_medians, masked_medians, masked_negative_medians = get_medians(mcmc_result, mcmc_result_negative, true_mus)

mask = true_avs < 0.15

fig, ax = plt.subplots(1,2, figsize=(8,4))
im = ax[0].scatter(true_mus, mcmc_result.point_estimates['mu'] - true_mus,
	c = true_avs, cmap='bwr',norm=TwoSlopeNorm(0.15))
negative_arr = (mcmc_result.point_estimates['mu'] - true_mus)[mask]
positive_arr = (mcmc_result.point_estimates['mu'] - true_mus)[~mask]

# for i in range(num_times_to_boostrap):
# 	ax[0].axhline(medians[i], color='dimgray', alpha = 0.1)
# 	ax[0].axhline(masked_medians[i], color='cornflowerblue', alpha = 0.1)
ax[0].axhline(np.median(negative_arr), linestyle='dashed', color='b', label='median residual (True. AV < 0.15)')
ax[0].axhline(np.median(positive_arr), linestyle='dashed', color='r', label='median residual (True. AV > 0.15)')

ax[0].axhline(np.median(mcmc_result.point_estimates['mu'] - true_mus), linestyle='dashed', color='k', label='median residual (all)')

plt.colorbar(im, label='True $A_V$')
ax[0].axhline(0, color='k', linewidth=0.5)
ax[0].set_title("AV Constrained to be positive")
ax[0].set_ylim(-0.5, 0.5)
ax[0].set_xlabel("True $\\mu$")
ax[0].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")
ax[0].legend()


im = ax[1].scatter(true_mus, mcmc_result_negative.point_estimates['mu'] - true_mus, 
	c = true_avs, cmap='bwr',norm=TwoSlopeNorm(0.15))

negative_arr = (mcmc_result_negative.point_estimates['mu'] - true_mus)[mask]
postive_arr = (mcmc_result_negative.point_estimates['mu'] - true_mus)[~mask]

# for i in range(num_times_to_boostrap):
# 	ax[1].axhline(negative_medians[i], color='dimgray', alpha = 0.1)
# 	ax[1].axhline(masked_negative_medians[i], color='cornflowerblue', alpha = 0.1)
ax[1].axhline(np.median(positive_arr), linestyle='dashed', color='r', label='median residual (True. AV > 0.15)')
ax[1].axhline(np.median(mcmc_result_negative.point_estimates['mu'] - true_mus), 
	linestyle='dashed', label='median residual (all)', color='k')
ax[1].axhline(np.median(negative_arr), linestyle='dashed', color='b', label='median residual (True. AV < 0.15)')

plt.colorbar(im, label='True $A_V$')
ax[1].set_title("AV not constrained to be positive")
ax[1].axhline(0, color='k', linewidth=0.5)
ax[1].set_ylim(-0.5, 0.5)
ax[1].set_xlabel("True $\\mu$")
ax[1].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")
ax[1].legend()
plt.show()
print(x)

fig, ax = plt.subplots(1,2, figsize=(8,4))
constrained_hr = mcmc_result.point_estimates['mu'] - true_mus
unconstrained_hr = mcmc_result_negative.point_estimates['mu'] - true_mus
av_bins = np.arange(0, 1.2, 0.05)
print(av_bins)
for i in range(len(av_bins) - 1):
	bin_center = (av_bins[i] + av_bins[i + 1]) / 2
	mask = (av_bins[i] < true_avs) & (true_avs < av_bins[i+1])
	print(bin_center, sum(mask))
	if sum(mask) > 0:
		ax[0].plot(bin_center, np.median(constrained_hr[mask]), 'bo')
		ax[1].plot(bin_center, np.median(unconstrained_hr[mask]), 'bo')

ax[0].set_xlabel("True $A_V$")
ax[1].set_xlabel("True $A_V$")

ax[0].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")
ax[1].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")

ax[1].set_title("AV not constrained to be positive")
ax[0].set_title("AV Constrained to be positive")
ax[0].axhline(0, color='k', linewidth=0.5)
ax[1].axhline(0, color='k', linewidth=0.5)
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,4))
constrained_hr = mcmc_result.point_estimates['mu'] - true_mus
unconstrained_hr = mcmc_result_negative.point_estimates['mu'] - true_mus
av_bins = np.arange(0, 1.2, 0.05)
print(av_bins)
for i in range(len(av_bins) - 1):
	bin_center = (av_bins[i] + av_bins[i + 1]) / 2
	mask = (av_bins[i] < true_avs) & (true_avs < av_bins[i+1])
	# print(bin_center, sum(mask))
	if sum(mask) > 0:
		ax.plot(bin_center, np.median(constrained_hr[mask]), 'bo')
		ax.plot(bin_center, np.median(unconstrained_hr[mask]), 'ro')

ax.plot(bin_center, np.median(constrained_hr[mask]), 'bo', label='constrained')
ax.plot(bin_center, np.median(unconstrained_hr[mask]), 'ro', label='unconstrained')
ax.set_xlabel("True $A_V$")

ax.set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")

ax.axhline(0, color='k', linewidth=0.5)
plt.legend()
plt.show()

# plt.hist(true_avs)
# plt.xlabel("True $A_V$")
# plt.show()
plt.plot(true_avs, mcmc_result_negative.point_estimates['AV'], 'o')
plt.xlabel("True $A_V$")
plt.ylabel("Unconstrained MCMC $A_V$")
plt.plot(np.linspace(-0.01, 1.3), np.linspace(-0.01, 1.3), c='k', linestyle='dashed', label="y=x")
plt.legend()
plt.axhline(0, color='k', linewidth=0.5)
plt.show()

print(x)

fig, ax = plt.subplots(1,2, figsize=(8,4))

ax[0].plot(true_avs, mcmc_result_negative.point_estimates['mu'] - true_mus, 'o')
ax[0].axhline(0, color='k')
# ax[0].set_ylim(-0.5, 0.5)
ax[0].set_xlabel("True $A_V$")
ax[0].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")

ax[1].plot(mcmc_result_negative.point_estimates['AV'], mcmc_result_negative.point_estimates['mu'] - true_mus, 'o')
ax[1].axhline(0, color='k')
ax[1].axvline(0, color='k')

# ax[1].set_ylim(-0.5, 0.5)
ax[1].set_xlabel("MCMC Fit $A_V$")
ax[1].set_ylabel("MCMC Fit $\\mu$ - True $\\mu$")
fig.suptitle("AV not constrained to be positive")
plt.show()


fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].plot(mcmc_result.point_estimates['AV'], mcmc_result_negative.point_estimates['mu'] - mcmc_result.point_estimates['mu'], 'o')
ax[0].set_xlabel("Constrained MCMC Fit $A_V$")
ax[0].set_ylabel("Unconstrained MCMC Fit $\\mu$ - Constrained MCMC Fit $\\mu$")
ax[0].axhline(0, color='k')
ax[0].axvline(0, color='k')


ax[1].plot(mcmc_result_negative.point_estimates['AV'], mcmc_result_negative.point_estimates['mu'] - mcmc_result.point_estimates['mu'], 'o')
ax[1].set_xlabel("Unconstrained MCMC Fit $A_V$")
ax[1].set_ylabel("Unconstrained MCMC Fit $\\mu$ - Constrained MCMC Fit $\\mu$")
ax[1].axhline(0, color='k')
ax[1].axvline(0, color='k')

plt.show()

plt.plot(mcmc_result_negative.point_estimates['AV'] - mcmc_result.point_estimates['AV'], mcmc_result_negative.point_estimates['mu'] - mcmc_result.point_estimates['mu'], 'o')
plt.xlabel("Unconstrained MCMC Fit $A_V$ - Constrained MCMC Fit $A_V$")
plt.ylabel("Unconstrained MCMC Fit $\\mu$ - Constrained MCMC Fit $\\mu$")
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.show()


plt.scatter(mcmc_result_negative.point_estimates['AV'], mcmc_result.point_estimates['AV'], c=np.log10(true_avs))
plt.plot(np.linspace(-0.3, 1.3), np.linspace(-0.3, 1.3), c='k', linestyle='dashed', label="y=x")
plt.xlabel("Unconstrained MCMC Fit $A_V$")
plt.ylabel("Constrained MCMC Fit $A_V$")
plt.axhline(0, color='k')
plt.axvline(0, color='k')
cbar = plt.colorbar()
cbar.set_label("log(True $A_V$)")
plt.legend()
plt.show()

print(sum(mcmc_result_negative.point_estimates['AV'] < 0))
print(np.where(np.all(mcmc_result_negative.av_samples < 0, axis=1)))
print(mcmc_result_negative.av_samples[66].shape)

plt.scatter(np.log10(mcmc_result_negative.point_estimates['AV']), np.log10(mcmc_result.point_estimates['AV']), c=np.log10(true_avs))
plt.plot(np.linspace(-3, 0.2), np.linspace(-3, 0.2), c='k', linestyle='dashed', label="y=x")
plt.xlabel("log Unconstrained MCMC Fit $A_V$")
plt.ylabel("log Constrained MCMC Fit $A_V$")
plt.axhline(0, color='k')
plt.axvline(0, color='k')
cbar = plt.colorbar()
cbar.set_label("log(True $A_V$)")
plt.legend()
plt.show()


