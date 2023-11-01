from bayesn_model import SEDmodel
import numpy as np
import os.path
from numpyro.infer import init_to_value
import jax.numpy as jnp


model = SEDmodel(load_model='T21_model')

dataset = 'sim_population_24'


# if os.path.exists("results/" + dataset):
# 	raise ValueError("It looks like this dataset has already been fit.")
# else:
# 	os.makedirs("results/" + dataset)

epsilons_on = True


filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

for i in range(50):
	print(i)

	sn_list = [int(i)]

	np.savetxt("temp_sn_list.txt", sn_list, fmt="%d", header="SNID", comments="")

	model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
	                      filt_map_dict, data_mode='flux', sn_list="temp_sn_list.txt")
	print("Epsilons on:", epsilons_on)
	print("Fitting MCMC...")
	model.fit(250, 250, 4, str(dataset) + "/" + str(i) + '_mcmc', epsilons_on=epsilons_on, chain_method='parallel', init_strategy='median')

	print("Fitting VI...")
	# model.fit_with_vi(str(dataset) + "/" + str(i)  + '_vi', init_strategy=init_to_value(values={'AV':jnp.array([0.01]), 'theta':jnp.array([1.]), 'Ds':jnp.array([35.])}))
	# model.fit_with_vi(str(dataset) + "/" + str(i)  + '_vi', init_strategy='median')
	model.fit_with_vi_laplace(str(dataset) + "/" + str(i)  + '_zltn', epsilons_on=epsilons_on, init_strategy='median')


	model.fit_with_vi_just_laplace(str(dataset) + "/" + str(i)  + '_just_laplace', epsilons_on=epsilons_on, init_strategy='median')
	
	model.fit_with_vi_laplace_multivariatenormal(str(dataset) + "/" + str(i)  + '_multivariatenormal', epsilons_on=epsilons_on, init_strategy='median')
