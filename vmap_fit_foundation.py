from bayesn_model import SEDmodel
import numpy as np
import os.path
from numpyro.infer import init_to_value
import jax
import jax.numpy as jnp
import pandas as pd
import timeit


model = SEDmodel(load_model='T21_model')

dataset = 'T21_training_set'

epsilons_on = True

filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])

sn_names = (sn_list.sn.values)

start = timeit.default_timer()


model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
                      filt_map_dict, data_mode='flux')
print(model.data.shape)
print(model.band_weights.shape)

# vi_laplace = jax.vmap(model.get_laplace_params_vmap, in_axes=(2, 0))
vi_laplace = jax.vmap(model.fit_zltn_vmap, in_axes=(2, 0))


# vi_map = jax.vmap(model.fit_with_vi_laplace_vmap("out", epsilons_on=True), in_axes=(2, 0))
samples_dict = vi_laplace(model.data, model.band_weights)

print(samples_dict)
print(samples_dict['AV'].shape)

end = timeit.default_timer()
print('Total time: ', end - start)

#np.save("foundation_vmap_112023.npy", samples_dict)
