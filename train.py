import numpy as np
import matplotlib.pyplot as plt
import lcdata
import pandas as pd
import parsnip
from model import Model
import pickle

dataset_path = 'data/bayesn_sim_test_z0_noext_units_25000.h5'
dataset = lcdata.read_hdf5(dataset_path)[:10]
bands = parsnip.get_bands(dataset)

param_path = 'data/bayesn_sim_test_z0_noext_units_25000_params.pkl'
params = pickle.load(open(param_path, 'rb'))
del params['epsilon']
params = pd.DataFrame(params)
pd_dataset = dataset.meta.to_pandas()
pd_dataset = pd_dataset.astype({'object_id': int})
params = params.merge(pd_dataset, on='object_id')
print(params.theta.values[0])

model = Model(bands, device='cpu')
result = model.fit(dataset)
print(result['theta'].shape)
print(result['theta'][:, :])
# plt.scatter(params.theta.values, result['theta'][0, :])
# plt.show()
