from zltn_utils import *
import numpy as np
import jax.numpy as jnp
import corner
import matplotlib.pyplot as plt
from jax.random import PRNGKey
from numpyro.infer import init_to_median


mean_vector = jnp.array([-0.1, 34., 2.])
test_cov = jnp.eye(3)

z = MultiZLTN(mean_vector, test_cov)
samples = z.sample(PRNGKey(321), (1000,))
print(samples)

print(len(samples[:,0]))


plt.hist(samples[:,0], histtype='step', label='ZLTN samples')
plt.hist(np.random.normal(-0.1, 1, size=1000), histtype='step', label='Normal samples')
plt.axvline(-0.1, color = 'r', linestyle='dashed', label='mean')
plt.legend()
plt.show()
# corner.corner(np.array(samples))
# plt.show()


prior_mus = jnp.array([-0.2, 0.9, 0.2])
prior_cov = jnp.array([[0.9, -0.7, -0.9], [-0.7, 1.0, 0.5], [0.5, -0.9, 1.0]])

def model_3d(data):
  prior_mus = jnp.array([-0.2, 0.9, 0.2])
  prior_cov = jnp.array([[0.9, -0.7, -0.9], [-0.7, 1.0, 0.5], [0.5, -0.9, 1.0]])
  # print(shape(prior_mus))
  Av = numpyro.sample("Av", MultiZLTN(prior_mus, prior_cov))

# guide_3d = AutoMultiZLTNGuide(model_3d, init_loc_fn=init_to_median())

# optimizer = Adam(0.01)
# svi = SVI(model_3d, guide_3d, optimizer, loss=Trace_ELBO(10))
# svi_result = svi.run(PRNGKey(123), 50000, data=None)
# params, losses = svi_result.params, svi_result.losses

# predictive = Predictive(guide_3d, params=params, num_samples=1000)
# samples = predictive(PRNGKey(123), data=None)
# print(samples)

# fig = corner.corner(np.array(samples['Av']))
# corner.corner(np.array(MultiZLTN(prior_mus, prior_cov).sample(PRNGKey(123),(1000,))), fig = fig, color='r')
# corner.overplot_lines(fig, vi_median, linestyle = 'dashed', color='g')

# plt.show()

# plt.hist(np.array(samples['Av'])[:,0], histtype='step')
# plt.hist(np.random.normal(-0.2, np.sqrt(0.9), size=1000), histtype='step')
# plt.show()


test_Avs = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]


for i in test_Avs:
	test_loc = jnp.array([i, 0.9, 0.2])
	test_cov = jnp.array([[0.9, -0.7, -0.9], [-0.7, 1.0, 0.5], [0.5, -0.9, 1.0]])
	test_samples = MultiZLTN(test_loc, test_cov).sample(PRNGKey(123), (1000,))

	fig = corner.corner(np.array(test_samples))
	corner.overplot_lines(fig, test_loc, linestyle = 'dashed', color='g')
	plt.show()


