from bayesn_model import SEDmodel
import numpy as np
import os.path
from numpyro.infer import init_to_value
from jax.random import PRNGKey, normal
import jax.numpy as jnp
import pandas as pd


model = SEDmodel(load_model='T21_model')

# dataset = 'Foundation_DR1'
# dataset = 'sim_nonzero_eps'

dataset = 'T21_training_set'

epsilons_on = True

# if os.path.exists("results/" + dataset):
# 	raise ValueError("It looks like this dataset has already been fit.")
# else:
# 	os.makedirs("results/" + dataset)


filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

# foundation_file = open("data/lcs/Foundation_DR1/Foundation_DR1/Foundation_DR1.LIST","r")
# s = 'Foundation_DR1_ASASSN-15fa.txt\n'

# sn_names = [s[15:-5] for s in foundation_file.readlines()]

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])

sn_names = (sn_list.sn.values)

best_ks = []
last_ks = []
best_samples_arr = []
last_samples_arr = []

def postprocess_add_mu(model, samples):
    # num_sn = samples['Ds'].shape[0]
    num_sn = 1
    samples['Ds'] = samples['Ds'].reshape((num_sn,5000))
    muhat = model.data[-3, 0, :]
    # print(muhat.shape)
    muhat_err = 10
    Ds_err = jnp.sqrt(muhat_err * muhat_err + model.sigma0 * model.sigma0)
    mu_mean = (np.squeeze(samples['Ds']) * jnp.power(muhat_err, 2) + muhat[...,None] * jnp.power(model.sigma0, 2)) / jnp.power(Ds_err, 2)

    mu_sigma = jnp.sqrt((jnp.power(model.sigma0, 2) * jnp.power(muhat_err, 2)) / jnp.power(Ds_err, 2))
    standard_normal_samples = normal(PRNGKey(123), shape=mu_mean.shape)
    mu = mu_mean + standard_normal_samples * mu_sigma
    # print(mu.shape)
    delM = np.squeeze(samples['Ds']) - mu
    samples['mu'] = mu
    samples['delM'] = delM
    return samples

for i, sn_name in enumerate(sn_names):
	print(i, sn_name)
	sn_list = [sn_name]

	np.savetxt("temp_sn_list.txt", sn_list, fmt="%s", header="SNID", comments="")

	# model.process_dataset('foundation', 'data/lcs/Foundation_DR1/Foundation_DR1/Foundation_DR1_' +sn_name + '.txt','data/lcs/Foundation_DR1/Foundation_DR1_' +sn_name + '.txt',
	#                       filt_map_dict, data_mode='flux')

	model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
	                      filt_map_dict, data_mode='flux', sn_list="temp_sn_list.txt")

	# print("Fitting MCMC...")
	# model.fit(250, 250, 4, str(dataset) + "/" + str(i) + '_mcmc', 
	# 	epsilons_on=epsilons_on, chain_method='parallel', 
	# 	init_strategy='median')

	print("Fitting VI...")
	# model.fit_with_vi(str(dataset) + "/" + str(i)  + '_vi', init_strategy=init_to_value(values={'AV':jnp.array([0.01]), 'theta':jnp.array([1.]), 'Ds':jnp.array([35.])}))
	# model.fit_with_vi(str(dataset) + "/" + str(i)  + '_vi', init_strategy='median')
	best_k, last_k, best_samples, last_samples = model.fit_zltn_get_ks(model.data, model.band_weights)
	
	best_samples = postprocess_add_mu(model, best_samples)
	last_samples = postprocess_add_mu(model, last_samples)

	best_ks.append(best_k)
	last_ks.append(last_k)
	best_samples_arr.append(best_samples)
	last_samples_arr.append(last_samples)

	print(best_ks, last_ks)
	# print(best_samples)

	np.savetxt("foundation_results/best_ks_032824.txt", np.array(best_ks))
	np.savetxt("foundation_results/last_ks_032824.txt", np.array(last_ks))
	np.savez("foundation_results/best_samples_032824", np.array(best_samples_arr))
	np.savez("foundation_results/last_samples_032824", np.array(last_samples_arr))


