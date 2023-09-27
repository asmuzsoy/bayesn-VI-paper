import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.cosmology import FlatLambdaCDM
import pickle
import numpy as np
import corner


mcmc_medians = {'AV':[], 'mu':[], 'theta':[]}
# vi_medians = {'AV':[], 'mu':[], 'theta':[]}
vi_modes = {'AV':[], 'mu':[], 'theta':[]}

mu_sample_diffs = []
av_samples = []
av_vi_samples = []

mu_samples = []
ds_samples = []
delm_samples = []

ds_sample_diffs = []
ds_mcmc_sample_diffs = []

ds_means = []
ds_errs = []

num_to_plot = 200

fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

dataset_number=16

sigma0 = 0.103
muhat_err = 10
from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')
dataset = 'sim_population_' + str(dataset_number)
filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

muhats = []
for i in range(num_to_plot):
	print(i)

	sn_list = [int(i)]

	np.savetxt("temp_sn_list.txt", sn_list, fmt="%d", header="SNID", comments="")

	model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
	                      filt_map_dict, data_mode='flux', sn_list="temp_sn_list.txt")

	muhat = model.data[-3, 0, :]
	muhats.append(muhat[0])





title_str = ""

if dataset_number == 6:
	title_str = "epsilons off for sims, on for fitting"
if dataset_number == 5 or dataset_number == 7:
	title_str = "no epsilons for sim/fitting"
if dataset_number == 4:
	title_str = "using epsilons for both sims and fitting"

true_avs = np.loadtxt("sim_population_AV_" + str(dataset_number) + ".txt")[:num_to_plot]
true_thetas = np.loadtxt("sim_population_theta_" + str(dataset_number) + ".txt")[:num_to_plot]
true_z = np.loadtxt("sim_population_z_" + str(dataset_number) + ".txt")[:num_to_plot]
true_mus = np.array([cosmo.distmod(z).value for z in true_z])

def get_mode_from_samples(samples):
	hist, bin_edges = np.histogram(samples, bins=50)
	max_index = np.argmax(hist)
	mode = (bin_edges[max_index] + bin_edges[max_index + 1])/2
	return mode


for i in range(num_to_plot):
	dataset = 'sim_population_' + str(dataset_number) + '/' + str(i)
	with (open("results/" + dataset + "_vi/chains.pkl", "rb")) as openfile:
		vi_objects = pickle.load(openfile)

	with (open("results/" + dataset + "_mcmc/chains.pkl", "rb")) as openfile:
		mcmc_objects = pickle.load(openfile)

	vi_params = np.load("results/" + dataset + "_vi/vi_params.npz")


	for var in ['AV', 'mu', 'theta']:
		mcmc_samples = mcmc_objects[var].reshape((1000,))
		vi_samples = np.squeeze(vi_objects[var])

		if var == "AV":
			# vi_modes[var].append(get_mode_from_samples(vi_samples))
			mcmc_medians[var].append(get_mode_from_samples(mcmc_samples))
			# vi_modes[var].append(np.maximum(0, vi_params['mu'][0][0])) # mu means mean here
			vi_modes[var].append(get_mode_from_samples(vi_samples)) # mu means mean here

			# vi_modes[var].append(vi_params['mu'][0][0]) # mu means mean here


			av_samples.append(mcmc_samples)


			# plt.hist(vi_samples, histtype='step',  color='k', label='VI')
			# plt.hist(mcmc_samples, histtype='step', color='r', label='MCMC')
			# plt.axvline(mode, linestyle='dashed', color='k', label='VI mode')
			# plt.axvline(np.median(mcmc_samples), linestyle='dashed', color='r', label='MCMC median')
			# plt.axvline(true_avs[i], linestyle='dashed', color='tab:green', label='True value')

			# plt.legend()
			# plt.show()
		elif var == 'theta':
			vi_modes[var].append(np.median(vi_samples))
			# vi_modes[var].append(vi_params['mu'][1][0]) # mu means mean here
			mcmc_medians[var].append(np.median(mcmc_samples))

		else: # mu
			vi_modes[var].append(np.median(vi_samples))
			# vi_modes[var].append(vi_params['mu'][-1][0])
			mcmc_medians[var].append(np.median(mcmc_samples))
			mu_sample_diffs.append(vi_samples - mcmc_samples)	
			ds_sample_diffs.append(vi_params['mu'][-1][0] - np.median(vi_samples))
			ds_mcmc_sample_diffs.append(vi_params['mu'][-1][0] - mcmc_samples)
			ds_means.append(vi_params['mu'][-1][0])
			ds_errs.append(vi_params['cov'][-1][-1])

	mu_samples.append(np.squeeze(vi_objects['mu']))
	ds_samples.append(np.squeeze(vi_objects['Ds']))
	delm_samples.append(np.squeeze(vi_objects['delM']))
	av_vi_samples.append(np.squeeze(vi_objects['AV']))


# plt.hist(vi_modes['AV'])
# plt.show()

mu_sample_diffs = np.array(mu_sample_diffs)
ds_mcmc_sample_diffs = np.array(ds_mcmc_sample_diffs)
delm_samples = np.array(delm_samples)
av_vi_samples = np.array(av_vi_samples)

# muhats = np.array(muhats)
ds_means = np.array(ds_means)
ds_errs = np.array(ds_errs)

print(ds_errs.shape)

av_samples = np.array(av_samples)

for var in ['AV', 'mu', 'theta']:
	mcmc_medians[var] = np.array(mcmc_medians[var])
	vi_modes[var] = np.array(vi_modes[var])


print(vi_modes['AV'])

fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(true_mus, vi_modes['mu'], 'o')
ax[0][0].plot(np.linspace(min(true_mus), max(true_mus)),np.linspace(min(true_mus), max(true_mus)))
ax[0][0].set_ylabel('Fit $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(true_mus, vi_modes['mu'] - true_mus, 'o')
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (fit - true)', fontsize = 16)
ax[1][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[0][1].plot(true_thetas, vi_modes['theta'], 'o')
ax[0][1].plot(np.linspace(min(true_thetas), max(true_thetas)),np.linspace(min(true_thetas), max(true_thetas)))
ax[0][1].set_ylabel('Fit $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(true_thetas, vi_modes['theta'] - true_thetas, 'o')
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (fit - true)', fontsize = 16)
ax[1][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[0][2].plot(true_avs, vi_modes['AV'], 'o')
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('Fit $A_V$', fontsize = 16)
ax[0][2].set_xlim(0,1)
ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(true_avs, vi_modes['AV'] - true_avs, 'o')
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (fit - true)', fontsize = 16)
ax[1][2].set_xlabel('True $A_V$', fontsize = 16)
# ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2,3, figsize = (16,9))
ax[0][0].plot(mcmc_medians['mu'], vi_modes['mu'], 'o')
ax[0][0].plot(np.linspace(min(mcmc_medians['mu']), max(mcmc_medians['mu'])),np.linspace(min(mcmc_medians['mu']), max(mcmc_medians['mu'])))
ax[0][0].set_ylabel('VI $\\mu$', fontsize = 16)
# ax[0][0].set_xlabel('True $\\mu$', fontsize = 16)


ax[1][0].plot(mcmc_medians['mu'], vi_modes['mu'] - mcmc_medians['mu'], 'o')
ax[1][0].axhline(0, color = 'k')
ax[1][0].set_ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
ax[1][0].set_xlabel('MCMC $\\mu$', fontsize = 16)


ax[0][1].plot(mcmc_medians['theta'], vi_modes['theta'], 'o')
ax[0][1].plot(np.linspace(min(mcmc_medians['theta']), max(mcmc_medians['theta'])),np.linspace(min(mcmc_medians['theta']), max(mcmc_medians['theta'])))
ax[0][1].set_ylabel('VI $\\theta$', fontsize = 16)
# ax[0][1].set_xlabel('True $\\theta$', fontsize = 16)

ax[1][1].plot(mcmc_medians['theta'], vi_modes['theta'] - mcmc_medians['theta'], 'o')
ax[1][1].axhline(0, color = 'k')
ax[1][1].set_ylabel('Residual $\\theta$ (VI - MCMC)', fontsize = 16)
ax[1][1].set_xlabel('MCMC $\\theta$', fontsize = 16)

ax[0][2].plot(mcmc_medians['AV'], vi_modes['AV'], 'o')
ax[0][2].plot(np.linspace(0, 2.1),np.linspace(0, 2.1))
ax[0][2].set_ylabel('VI $A_V$', fontsize = 16)
# ax[0][2].set_xlim(0,1)
# ax[0][2].set_ylim(0,1)

# ax[0][2].set_xlabel('True $A_V$', fontsize = 16)

ax[1][2].plot(mcmc_medians['AV'], vi_modes['AV'] - mcmc_medians['AV'], 'o')
ax[1][2].axhline(0, color = 'k')
ax[1][2].set_ylabel('Residual $A_V$ (VI - MCMC)', fontsize = 16)
ax[1][2].set_xlabel('MCMC $A_V$', fontsize = 16)
# ax[1][2].set_xlim(0,1)

for axis in ax.flatten():
  axis.tick_params(axis='x', labelsize=12)
  axis.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

plt.plot(mcmc_medians['AV'], vi_modes['mu'] - mcmc_medians['mu'], 'o')
plt.xlabel('MCMC $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
plt.title(title_str)
plt.show()


stephen_values = (ds_means*muhat_err**2 + muhats*(sigma0**2 + ds_errs**2)) / (ds_errs**2 + sigma0**2 + muhat_err**2)
print((stephen_values).shape)

plt.plot(vi_modes['mu'], stephen_values, 'o')
plt.show()


plt.plot(mcmc_medians['AV'], stephen_values - mcmc_medians['mu'], 'o')
plt.xlabel('MCMC $A_V$', fontsize = 16)
plt.ylabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
plt.title(title_str + "- Stephen value")
plt.show()


plt.hist(vi_modes['mu'] - mcmc_medians['mu'])
plt.xlabel('Residual $\\mu$ (VI - MCMC)', fontsize = 16)
plt.show()

low_av_mu_diffs = ds_mcmc_sample_diffs.flatten()[av_samples.flatten() < 0.2]
print(len(low_av_mu_diffs))
plt.hist(low_av_mu_diffs, bins=20)
plt.axvline(0, linestyle='dashed', color='k')
plt.axvline(np.median(low_av_mu_diffs), linestyle='dashed', color='r', label='median diff')

plt.xlabel("$\\mu$ residual (VI - MCMC)")
plt.legend()
plt.show()

plt.hist(delm_samples.flatten())
plt.axvline(np.median(delm_samples), linestyle='dashed', color='r', label='median diff')
plt.xlabel("DelM")
plt.show()


fig, ax = plt.subplots(1,2)

ax[0].plot(vi_modes['AV'], [get_mode_from_samples(i) for i in ds_samples] - vi_modes['mu'], 'o')
ax[0].set_ylabel("VI Ds samples mode - VI Ds mean")
ax[0].set_xlabel("VI AV mean")


ax[1].plot(vi_modes['AV'], [np.median(i) for i in ds_samples] - vi_modes['mu'], 'o')
ax[1].set_ylabel("VI Ds samples median - VI Ds mean")
ax[1].set_xlabel("VI AV mean")


plt.show()


# for index_to_show in range(50):
# 	plt.hist(ds_samples[index_to_show])
# 	median_ds_residual = np.median(ds_samples[index_to_show] - vi_modes['mu'][index_to_show])
# 	plt.axvline(vi_modes['mu'][index_to_show], color='r')
# 	# plt.axvline(0, color='k')

# 	plt.title("Ds samples-VI mean, " + str(round(median_ds_residual,4)) + ","+ str(vi_modes['AV'][index_to_show]))
# 	plt.show()

# for index_to_show in range(50):
# 	plt.hist(av_vi_samples[index_to_show])
# 	plt.axvline(vi_modes['AV'][index_to_show], color='r')
# 	plt.show()


# print(true_avs[index_to_show])
# for index_to_show in range(100):
	
# 	plt.hist(mu_samples[index_to_show], histtype='step', label='mu', density = True)
# 	plt.axvline(mcmc_medians['mu'][index_to_show], color='tab:blue', label='MCMC median mu')

# 	plt.axvline(vi_modes['mu'][index_to_show], color='tab:orange', label='Variational Ds mean')

# 	plt.hist(ds_samples[index_to_show], histtype='step', label='Ds', density=True)

# 	plt.axvline(true_mus[index_to_show], color='k', linestyle='dashed', label='true mu value')
# 	plt.title("$\\mu$ residual (VI - MCMC): " + str(round((vi_modes['mu'] - mcmc_medians['mu'])[index_to_show],3	)))
# 	plt.legend()
# 	plt.show()

# plt.hist(delm_samples[index_to_show], histtype='step', label='delM')
# plt.legend()
# plt.show()

plt.hist(ds_sample_diffs)
plt.xlabel("Variational Ds mean - median of VI mu samples")
plt.show()
