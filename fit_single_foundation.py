from bayesn_model import SEDmodel
import numpy as np
import os.path
from numpyro.infer import init_to_value
import jax.numpy as jnp
import pandas as pd


model = SEDmodel(load_model='T21_model')

dataset = 'T21_training_set'

epsilons_on = True

filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

# sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])

# sn_names = (sn_list.sn.values)

sn_name = 'AT2016cyt'

sn_list = [sn_name]

np.savetxt("temp_sn_list.txt", sn_list, fmt="%s", header="SNID", comments="")

# model.process_dataset('foundation', 'data/lcs/Foundation_DR1/Foundation_DR1/Foundation_DR1_' +sn_name + '.txt','data/lcs/Foundation_DR1/Foundation_DR1_' +sn_name + '.txt',
#                       filt_map_dict, data_mode='flux')

model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
                      filt_map_dict, data_mode='flux', sn_list="temp_sn_list.txt")

# print("Fitting MCMC...")
# model.fit(250, 250, 4, str(sn_name) + '_mcmc', 
#     epsilons_on=epsilons_on, chain_method='parallel', 
#     init_strategy='median')

print("Fitting VI...")
# model.fit_with_vi(str(dataset) + "/" + str(i)  + '_vi', init_strategy=init_to_value(values={'AV':jnp.array([0.01]), 'theta':jnp.array([1.]), 'Ds':jnp.array([35.])}))
# model.fit_with_vi(str(dataset) + "/" + str(i)  + '_vi', init_strategy='median')
model.fit_with_vi_laplace(str(sn_name) + '_vi', 
    epsilons_on=epsilons_on, init_strategy='median')