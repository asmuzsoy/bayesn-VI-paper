import numpy as np
import lcdata
import parsnip
from model import Model

dataset_path = 'data/bayesn_sim_test_z0_noext_25000.h5'
dataset = lcdata.read_hdf5(dataset_path)[:100]
bands = parsnip.get_bands(dataset)

model = Model(bands)
model.fit(dataset)
t = 48
band_indices = np.array([[i] * int(t/4) for i in range(4)])
model.get_flux(0, band_indices)
