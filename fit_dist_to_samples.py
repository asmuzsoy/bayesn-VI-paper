import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_sample, init_to_value, Predictive
import numpyro.distributions as dist
from numpyro.optim import Adam, ClippedAdam
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoDiagonalNormal
from numpyro.distributions.transforms import LowerCholeskyAffine
from numpyro.primitives import plate
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import jax
from jax import device_put
import jax.numpy as jnp
from jax.random import PRNGKey, split
import spline_utils
from zltn_utils import *
import pickle
from astropy.cosmology import FlatLambdaCDM


fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

dataset = 'sim_population_12/12'
dataset_number = 12
true_av = np.loadtxt("sim_population_AV_" + str(dataset_number) + ".txt")[12]
true_theta = np.loadtxt("sim_population_theta_" + str(dataset_number) + ".txt")[12]
true_z = np.loadtxt("sim_population_z_" + str(dataset_number) + ".txt")[12]
true_mu = cosmo.distmod(true_z).value



def fit_model(samples):

	mean_vector = numpyro.sample("mean_vector", dist.MultivariateNormal(jnp.array([true_av,true_mu,true_theta]), jnp.eye(3)))
	# test_cov = jnp.eye(3)
	# print(test_cov.shape)
	# code from https://num.pyro.ai/en/latest/distributions.html
	d = 3
	std_vector = numpyro.sample("std_vector", dist.HalfNormal(jnp.ones(d)))
	# concentration = jnp.ones(3)  # Implies a uniform distribution over correlation matrices
	corr_mat = numpyro.sample("corr_mat", dist.LKJ(d))
	sigma = jnp.sqrt(std_vector)
	cov_mat = jnp.matmul(jnp.matmul(jnp.diag(sigma), corr_mat), jnp.diag(sigma))
	# cov_mat = jnp.outer(std_vector, std_vector) * corr_mat



	with numpyro.plate("observations", len(samples)):
	# obs = np.zeros((200,3))
	# for i in range(200):
	# 	print(i)
		# print(mean_vector.shape)
		# print(cov_mat.shape)
		# print(samples.shape)
		# obs[i] = numpyro.sample("obs" + str(i), MultiZLTN(mean_vector, covariance_matrix=cov_mat), obs=samples[i])
		obs = numpyro.sample("obs", MultiZLTN(mean_vector, covariance_matrix=cov_mat), obs=samples)



with (open("results/" + dataset + "_mcmc/chains.pkl", "rb")) as openfile:
	mcmc_objects = pickle.load(openfile)


for i in range(1):
	mcmc_results = []
	for var in ['AV', 'mu', 'theta']:
		mcmc_samples = mcmc_objects[var][:,:,i].reshape((1000,))

		mcmc_results.append(mcmc_samples)

	mcmc_results = np.array(mcmc_results).T

print(mcmc_results)



optimizer = Adam(0.001)

guide = AutoDelta(fit_model)
svi = SVI(fit_model, guide, optimizer, loss=Trace_ELBO(10))
svi_result = svi.run(PRNGKey(123), 50000, mcmc_results)

params, losses = svi_result.params, svi_result.losses
# predictive = Predictive(guide, params=params, num_samples=1000)
# samples = predictive(PRNGKey(123), data=None)
# print(samples.keys())

vi_median = guide.median(params)['mean_vector']
vi_std_vector = guide.median(params)['std_vector']
vi_corr_mat = guide.median(params)['corr_mat']

vi_sigma = jnp.sqrt(vi_std_vector)
vi_cov_mat = jnp.matmul(jnp.matmul(jnp.diag(vi_sigma), vi_corr_mat), jnp.diag(vi_sigma))


print(vi_median)
print(vi_cov_mat)
range1 = [(-1,0.2), (33.5, 35), (-0,1.5)]

fig = corner.corner(mcmc_results, color = 'r', range = range1)
corner.corner(np.array(MultiZLTN(vi_median, vi_cov_mat).sample(PRNGKey(123),(1000,))), fig=fig, color = 'k', range = range1)
# corner.corner(np.array(samples), fig=fig, color = 'k', range = range1)

corner.overplot_lines(fig, vi_median, linestyle = 'dashed', color='g')
colors = ['r', 'g']

labels = ['MCMC Samples','VI parameters']

plt.legend(
    handles=[
        mlines.Line2D([], [], color=colors[i], label=labels[i])
        for i in range(len(labels))
    ],
    fontsize=14, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
)

plt.show()

