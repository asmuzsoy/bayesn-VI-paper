import matplotlib.pyplot as plt
import pickle
import numpy as np
import corner


with (open("results/T21_fit_1_vi/chains.pkl", "rb")) as openfile:
	vi_objects = pickle.load(openfile)

with (open("results/T21_fit_1_mcmc/chains.pkl", "rb")) as openfile:
	mcmc_objects = pickle.load(openfile)


print(vi_objects.keys())
print(vi_objects['AV'][:,0].shape)
print(vi_objects['AV'][:,1].shape)
print(vi_objects['AV'].shape)
print(vi_objects['AV'][0][0].shape)

# print(vi_objects['AV'])




print(mcmc_objects['AV'][:,:,0].shape)
# print(objects['AV'].shape)

for i in range(5):
	for var in ['AV', 'mu', 'theta']:
		mcmc_samples = mcmc_objects[var][:,:,i].reshape((1000,))
		vi_samples = np.squeeze(vi_objects[var][:,i])

		plt.hist(vi_samples, label = 'VI', histtype='step', density=True)
		plt.hist(mcmc_samples, label = 'MCMC', histtype='step', density=True)
		plt.xlabel(var)
		plt.legend()
		plt.show()
