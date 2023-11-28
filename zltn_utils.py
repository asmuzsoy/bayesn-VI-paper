import numpyro
from numpyro.infer.autoguide import AutoContinuous
from numpyro.distributions import constraints
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.distributions.distribution import Distribution 
from numpyro.distributions.continuous import MultivariateNormal, Normal
from numpyro.distributions.truncated import TruncatedNormal
from numpyro.distributions.constraints import _SingletonConstraint
from numpyro.distributions.transforms import biject_to, IdentityTransform, LowerCholeskyAffine
import torch
import jax.numpy as jnp
from numpyro.distributions.util import (
    is_prng_key,
    lazy_property,
    matrix_to_tril_vec,
    promote_shapes,
    signed_stick_breaking_tril,
    validate_sample,
    vec_to_tril_matrix,
)
from jax import lax, vmap
import matplotlib.pyplot as plt
from jax.random import PRNGKey, normal, exponential
import corner
import numpy as np
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoDiagonalNormal

key = PRNGKey(123)

# zltn = TruncatedNormal(loc = 0.01, scale = 0.005, low = 0)
# plt.hist(zltn.sample(PRNGKey(123), (1000,)))
# plt.show()


class _FirstPositive(_SingletonConstraint):
    event_dim = 1

    def __call__(self, x):
        print("Constraint is being called")
        return (x[0] >= 0)


    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)

firstpositive = _FirstPositive()

@biject_to.register(firstpositive)
def _transform_to_real(constraint):
    return IdentityTransform()

class MultiZLTN(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    reparametrized_params = [
        "loc",
        "covariance_matrix",
        "precision_matrix",
        "scale_tril",
    ]

    @constraints.dependent_property
    def support(self):
        return firstpositive

    def __init__(
        self,
        loc=0.0,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        print("initializing ZLTN")
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]
        if covariance_matrix is not None:
            loc, self.covariance_matrix = promote_shapes(loc, covariance_matrix)
            self.scale_tril = jnp.linalg.cholesky(self.covariance_matrix)
        elif precision_matrix is not None:
            loc, self.precision_matrix = promote_shapes(loc, precision_matrix)
            self.scale_tril = cholesky_of_inverse(self.precision_matrix)
        elif scale_tril is not None:
            loc, self.scale_tril = promote_shapes(loc, scale_tril)
            self.covariance_matrix = jnp.matmul(self.scale_tril, self.scale_tril.T)
        else:
            raise ValueError(
                "One of `covariance_matrix`, `precision_matrix`, `scale_tril`"
                " must be specified."
            )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2], jnp.shape(self.scale_tril)[:-2]
        )
        event_shape = jnp.shape(self.scale_tril)[-1:]
        # self.loc = loc[..., 0]
        self.mu = loc
        self.mu_1 = self.mu[0]
        print("Distribution mean:", self.mu_1)
        self.mu_2 = jnp.squeeze(self.mu[1:])
        print("input mu 2", self.mu_2)

        self.sigma = self.covariance_matrix
        self.sigma_11 = self.sigma[0][0]
        self.sigma_22 = self.sigma[1:, 1:]
        self.sigma_21 = self.sigma[1:, :1]
        self.sigma_12 = self.sigma[:1, 1:]


        print(loc)
        self.zltn_dist = TruncatedNormal(self.mu_1, self.scale_tril[0][0], low = 0.)
        print("batch shape: ", batch_shape)
        super(MultiZLTN, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        first_samples = self.zltn_dist.sample(key, sample_shape)
        sigma_11_inverse = 1./self.sigma_11

        if sample_shape==():
            mu_hat = self.mu_2 + jnp.matmul(self.sigma_21, ((1.0 / self.sigma_11) * (first_samples - self.mu_1)).T)
            sigma_hat = self.sigma_22 - jnp.matmul((self.sigma_21 * sigma_11_inverse), self.sigma_12)
            sigma_hat += 1.e-9 * jnp.eye(self.event_shape[0]-1)

            eps = normal(
                key, shape=sample_shape + self.batch_shape + (self.event_shape[0]-1,)
            )
            second_samples = mu_hat.T + jnp.squeeze(
                jnp.matmul(jnp.linalg.cholesky(sigma_hat), eps[..., jnp.newaxis]), axis=-1
            )
            return jnp.concatenate((first_samples,second_samples), axis=0)

        mu_hat = self.mu_2[:,None] + jnp.matmul(self.sigma_21, ((1.0 / self.sigma_11) * (first_samples - self.mu_1)).T)
        print("mu hat", mu_hat)
        sigma_hat = self.sigma_22 - jnp.matmul((self.sigma_21 * sigma_11_inverse), self.sigma_12)
        sigma_hat += 1.e-9 * jnp.eye(self.event_shape[0]-1)

        eps = normal(
            key, shape=sample_shape + self.batch_shape + (self.event_shape[0]-1,)
        )
        second_samples = mu_hat.T + jnp.squeeze(
            jnp.matmul(jnp.linalg.cholesky(sigma_hat), eps[..., jnp.newaxis]), axis=-1
        )
        return jnp.concatenate((first_samples,second_samples), axis=1)


    def log_prob(self, value):
        log_prob_x = self.zltn_dist.log_prob(value[...,0])
        sigma_11_inverse = 1. / self.sigma_11
        # mu_hat = self.mu_2 + self.sigma_21*sigma_11_inverse*(value[0]- self.mu_1)
        # print(self.mu_1.shape)
        # print(value)
        # print(value[...,0].shape)
        # print(value[0][0].shape)

        mu_hat = self.mu_2[...,None] + jnp.matmul(self.sigma_21, sigma_11_inverse * (value[...,0] - self.mu_1[None,...]))
        sigma_hat = self.sigma_22 - jnp.matmul((self.sigma_21 * sigma_11_inverse), self.sigma_12)
        # print(mu_hat.shape, sigma_hat.shape, value[...,1:].shape)
        m = MultivariateNormal(mu_hat.T, sigma_hat)

        log_prob_y_given_x = MultivariateNormal(mu_hat.T, sigma_hat).log_prob(value[...,1:])
        # print(log_prob_x + log_prob_y_given_x)
        return log_prob_x + log_prob_y_given_x

    # old method without vectorization
    # def log_prob(self, value):
    #     log_prob_x = self.zltn_dist.log_prob(value[0])
    #     sigma_11_inverse = 1. / self.sigma_11
    #     # mu_hat = self.mu_2 + self.sigma_21*sigma_11_inverse*(value[0]- self.mu_1)
    #     mu_hat = self.mu_2 + jnp.matmul(self.sigma_21, sigma_11_inverse * (value[0] - self.mu_1))
    #     sigma_hat = self.sigma_22 - jnp.matmul((self.sigma_21 * sigma_11_inverse), self.sigma_12)
    #     log_prob_y_given_x = MultivariateNormal(mu_hat, sigma_hat).log_prob(value[1:])
    #     print(log_prob_x + log_prob_y_given_x)
    #     return log_prob_x + log_prob_y_given_x

class AutoMultiZLTNGuide(AutoContinuous):

    # scale_tril_constraint = constraints.scaled_unit_lower_cholesky
    scale_tril_constraint = constraints.lower_cholesky


    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_median,
        init_scale=0.1,
    ):
        print("Initializing guide...")
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _get_posterior(self):
        print("getting posterior")
        loc = numpyro.param("{}_loc".format(self.prefix), self._init_latent)
        scale_tril = numpyro.param(
            "{}_scale_tril".format(self.prefix),
            jnp.identity(self.latent_dim) * self._init_scale,
            constraint=self.scale_tril_constraint,
        )
        return MultiZLTN(loc, scale_tril=scale_tril)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)


    def get_transform(self, params):
        print("getting posterior2")

        loc = params["{}_loc".format(self.prefix)]
        scale_tril = params["{}_scale_tril".format(self.prefix)]
        return LowerCholeskyAffine(loc, scale_tril)


    def get_posterior(self, params):
        """
        Returns a Multi ZLTN posterior distribution.
        """
        print("getting posterior2")

        transform = self.get_transform(params)
        return MultiZLTN(transform.loc, scale_tril=transform.scale_tril)


class My_Exponential(Distribution):
    reparametrized_params = ["rate"]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.real

    def __init__(self, rate=1.0, *, validate_args=None):
        self.rate = rate
        super(My_Exponential, self).__init__(
            batch_shape=jnp.shape(rate), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return (
            exponential(key, shape=sample_shape + self.batch_shape) / self.rate
        )


    @validate_sample
    def log_prob(self, value):
        return jnp.log(self.rate) - self.rate * value

    @property
    def mean(self):
        return jnp.reciprocal(self.rate)

    @property
    def variance(self):
        return jnp.reciprocal(self.rate**2)

    def cdf(self, value):
        return -jnp.expm1(-self.rate * value)

    def icdf(self, q):
        return -jnp.log1p(-q) / self.rate

# optimizer = Adam(0.01)

# guide = AutoMultiZLTNGuide(model_3d)
# svi = SVI(model_3d, guide, optimizer, loss=Trace_ELBO(num_particles = 10))
# svi_result = svi.run(key, 10000, None)
# params, losses = svi_result.params, svi_result.losses
# # print(params)
# predictive = Predictive(guide, params=params, num_samples=1000)
# posterior_samples = predictive(PRNGKey(123), None)

# prior_mus = jnp.array([0.2, 0.9, 3.0])
# prior_cov = jnp.array([[0.9, -0.7, -0.9], [-0.7, 1.0, 0.5], [0.5, -0.9, 1.0]])
# prior_samples = MultiZLTN(prior_mus, prior_cov).sample(key, (1000,))

# figure = corner.corner(np.array(posterior_samples['Av']))
# corner.corner(np.array(prior_samples), color = 'r', fig = figure)

# plt.show()


    # def median(self, params):
    #     loc = params["{}_loc".format(self.prefix)]
    #     return self._unpack_and_constrain(loc, params)


    # def quantiles(self, params, quantiles):
    #     transform = self.get_transform(params)
    #     quantiles = jnp.array(quantiles)[..., None]
    #     latent = dist.Normal(transform.loc, jnp.diagonal(transform.scale_tril)).icdf(
    #         quantiles
    #     )
    #     return self._unpack_and_constrain(latent, params)