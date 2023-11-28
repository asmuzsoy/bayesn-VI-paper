from bayesn_model import SEDmodel
import numpy as np
import os.path
from numpyro.infer import init_to_value
import jax
import jax.numpy as jnp
import pandas as pd
import timeit


model = SEDmodel(load_model='T21_model')

dataset = 'sim_population_26'
	
filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}


model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
	                      filt_map_dict, data_mode='flux')

print(model.data.shape)
print(model.band_weights.shape)

start = timeit.default_timer()

mcmc = jax.vmap(model.fit_mcmc_vmap, in_axes=(2, 0))
mcmc_samples= mcmc(model.data, model.band_weights)
np.save("low_av_mcmc_112023.npy", mcmc_samples)

t1 = timeit.default_timer()
print('MCMC: ', t1 - start)


laplace = jax.vmap(model.fit_laplace_vmap, in_axes=(2, 0))
laplace_samples= laplace(model.data, model.band_weights)
np.save("low_av_laplace_112023.npy", laplace_samples)

t2 = timeit.default_timer()
print('Laplace: ', t2 - t1)

zltn = jax.vmap(model.fit_zltn_vmap, in_axes=(2, 0))
zltn_samples= zltn(model.data, model.band_weights)
np.save("low_av_zltn_112023.npy", zltn_samples)

t3 = timeit.default_timer()
print('ZLTN: ', t3 - t2)

multinormal = jax.vmap(model.fit_multivariatenormal_vmap, in_axes=(2, 0))
multinormal_samples= multinormal(model.data, model.band_weights)
np.save("low_av_multinormal_112023.npy", multinormal_samples)

end = timeit.default_timer()
print('MultivariateNormal: ', end - t3)

print('Total time: ', end - start)


