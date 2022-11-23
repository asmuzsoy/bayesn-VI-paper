import numpy as np
import matplotlib.pyplot as plt
import lcdata
import pandas as pd
import parsnip
from model import Model
import pickle

dataset_path = 'data/bayesn_sim_test_z0_noext_thetarange_20.h5'
dataset = lcdata.read_hdf5(dataset_path)
bands = parsnip.get_bands(dataset)

param_path = 'data/bayesn_sim_test_z0_noext_thetarange_20_params.pkl'
params = pickle.load(open(param_path, 'rb'))
del params['epsilon']
params = pd.DataFrame(params)

pd_dataset = dataset.meta.to_pandas()
pd_dataset = pd_dataset.astype({'object_id': int})
print(pd_dataset)
inds = pd_dataset.object_id.values - 1
params = params.merge(pd_dataset, on='object_id')
print(params.theta.values[inds])
print(params)
raise ValueError('Nope')
print('Actual', params.theta.values)

model = Model(bands, device='cpu')
model.compare_gen_theta(dataset, params)
#result = model.fit(dataset)
#print(result['theta'])
#print(np.mean(result['theta'].numpy(), axis=0), np.std(result['theta'].numpy(), axis=0))
# plt.scatter(params.theta.values, result['theta'][0, :])
# plt.show()
