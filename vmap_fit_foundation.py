from bayesn_model import SEDmodel
import numpy as np
import os.path
from numpyro.infer import init_to_value
import jax
from jax.random import PRNGKey, normal
import jax.numpy as jnp
import pandas as pd
import timeit


model = SEDmodel(load_model='T21_model')

dataset = 'T21_training_set'

output_title = 'foundation'
todays_date = '122923'

epsilons_on = True

filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])

sn_names = (sn_list.sn.values)

start = timeit.default_timer()


model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
                      filt_map_dict, data_mode='flux')


def postprocess_add_mu(model, samples):
    num_sn = samples['Ds'].shape[0]
    samples['Ds'] = samples['Ds'].reshape((num_sn,1000))
    muhat = model.data[-3, 0, :]
    print(muhat.shape)
    muhat_err = 10
    Ds_err = jnp.sqrt(muhat_err * muhat_err + model.sigma0 * model.sigma0)
    mu_mean = (np.squeeze(samples['Ds']) * jnp.power(muhat_err, 2) + muhat[...,None] * jnp.power(model.sigma0, 2)) / jnp.power(Ds_err, 2)

    mu_sigma = jnp.sqrt((jnp.power(model.sigma0, 2) * jnp.power(muhat_err, 2)) / jnp.power(Ds_err, 2))
    standard_normal_samples = normal(PRNGKey(123), shape=mu_mean.shape)
    mu = mu_mean + standard_normal_samples * mu_sigma
    print(mu.shape)
    delM = np.squeeze(samples['Ds']) - mu
    samples['mu'] = mu
    samples['delM'] = delM
    return samples

start = timeit.default_timer()

## rewrite these to use methods
mcmc = jax.vmap(model.fit_mcmc_vmap, in_axes=(2, 0))
mcmc_samples = mcmc(model.data, model.band_weights)
mcmc_samples = postprocess_add_mu(model, mcmc_samples)
np.save("foundation_vmap_mcmc_" + todays_date + ".npy", mcmc_samples)

t1 = timeit.default_timer()
print('MCMC: ', t1 - start)


laplace = jax.vmap(model.fit_laplace_vmap, in_axes=(2, 0))
laplace_samples= laplace(model.data, model.band_weights)
laplace_samples = postprocess_add_mu(model, laplace_samples)
np.save("foundation_vmap_laplace_" + todays_date + ".npy", laplace_samples)


t2 = timeit.default_timer()
print('Laplace: ', t2 - t1)

zltn = jax.vmap(model.fit_zltn_vmap, in_axes=(2, 0))
zltn_samples= zltn(model.data, model.band_weights)
zltn_samples = postprocess_add_mu(model, zltn_samples)
np.save("foundation_vmap_zltn_" + todays_date + ".npy", zltn_samples)

t3 = timeit.default_timer()
print('ZLTN: ', t3 - t2)

multinormal = jax.vmap(model.fit_multivariatenormal_vmap, in_axes=(2, 0))
multinormal_samples= multinormal(model.data, model.band_weights)
multinormal_samples = postprocess_add_mu(model, multinormal_samples)
np.save("foundation_vmap_multinormal_" + todays_date + ".npy", multinormal_samples)

end = timeit.default_timer()
print('MultivariateNormal: ', end - t3)

print('Total time: ', end - start)

