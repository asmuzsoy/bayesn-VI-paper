import os
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, HMC, init_to_median
import numpyro.distributions as dist
import h5py
import lcdata
import sncosmo
from settings import parse_settings
import spline_utils
import time
import pickle
import pandas as pd
import jax
from jax import device_put
import jax.numpy as jnp
from jax.random import PRNGKey
import extinction
import yaml
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})

# jax.config.update('jax_platform_name', 'cpu')

print(jax.devices())


class Model(object):
    def __init__(self, bands, ignore_unknown_settings=False, settings={}, device='cuda',
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}):
        self.data = None
        self.device = device
        self.settings = parse_settings(bands, settings,
                                       ignore_unknown_settings=ignore_unknown_settings)
        self.M0 = -19.5
        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.sigma_pec = 150 / 3e5

        self.l_knots = np.genfromtxt('model_files/T21_model/l_knots.txt')
        self.tau_knots = np.genfromtxt('model_files/T21_model/tau_knots.txt')
        self.W0 = np.genfromtxt('model_files/T21_model/W0.txt')
        self.W1 = np.genfromtxt('model_files/T21_model/W1.txt')
        self.L_Sigma = np.genfromtxt('model_files/T21_model/L_Sigma_epsilon.txt')
        self.l_knots = device_put(self.l_knots)
        self.tau_knots = device_put(self.tau_knots)
        self.W0 = device_put(self.W0)
        self.W1 = device_put(self.W1)
        self.L_Sigma = device_put(self.L_Sigma)
        self.band_dict = {band: i for i, band in enumerate(bands)}

        self._setup_band_weights()

        KD_l = spline_utils.invKD_irr(self.l_knots)
        self.J_l_T = device_put(spline_utils.spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        self.KD_t = device_put(spline_utils.invKD_irr(self.tau_knots))
        self.load_hsiao_template()

        self.ZPT = 27.5
        self.J_l_T = device_put(self.J_l_T)
        self.hsiao_flux = device_put(self.hsiao_flux)
        self.J_l_T_hsiao = device_put(self.J_l_T_hsiao)
        self.xk = jnp.array(
            [0.0, 1e4 / 26500., 1e4 / 12200., 1e4 / 6000., 1e4 / 5470., 1e4 / 4670., 1e4 / 4110., 1e4 / 2700.,
             1e4 / 2600.])
        KD_x = spline_utils.invKD_irr(self.xk)
        self.M_fitz_block = device_put(spline_utils.spline_coeffs_irr(1e4 / self.model_wave, self.xk, KD_x))

    def load_hsiao_template(self):
        with h5py.File(os.path.join('data', 'hsiao.h5'), 'r') as file:
            data = file['default']

            hsiao_phase = data['phase'][()].astype('float64')
            hsiao_wave = data['wave'][()].astype('float64')
            hsiao_flux = data['flux'][()].astype('float64')

        KD_l_hsiao = spline_utils.invKD_irr(hsiao_wave)
        self.KD_t_hsiao = spline_utils.invKD_irr(hsiao_phase)
        self.J_l_T_hsiao = device_put(spline_utils.spline_coeffs_irr(self.model_wave,
                                                                           hsiao_wave, KD_l_hsiao))

        self.hsiao_t = device_put(hsiao_phase)
        self.hsiao_l = device_put(hsiao_wave)
        self.hsiao_flux = device_put(hsiao_flux.T)

    def _setup_band_weights(self):
        """Setup the interpolation for the band weights used for photometry"""
        # Build the model in log wavelength
        model_log_wave = np.linspace(np.log10(self.settings['min_wave']),
                                     np.log10(self.settings['max_wave']),
                                     self.settings['spectrum_bins'])

        model_spacing = model_log_wave[1] - model_log_wave[0]

        band_spacing = model_spacing / self.settings['band_oversampling']
        band_max_log_wave = (
                np.log10(self.settings['max_wave'] * (1 + self.settings['max_redshift']))
                + band_spacing
        )

        # Oversampling must be odd.
        assert self.settings['band_oversampling'] % 2 == 1
        pad = (self.settings['band_oversampling'] - 1) // 2
        band_log_wave = np.arange(np.log10(self.settings['min_wave']),
                                  band_max_log_wave, band_spacing)
        band_wave = 10 ** (band_log_wave)
        band_pad_log_wave = np.arange(
            np.log10(self.settings['min_wave']) - band_spacing * pad,
            band_max_log_wave + band_spacing * pad,
            band_spacing
        )
        band_pad_dwave = (
                10 ** (band_pad_log_wave + band_spacing / 2.)
                - 10 ** (band_pad_log_wave - band_spacing / 2.)
        )

        ref = sncosmo.get_magsystem(self.settings['magsys'])

        band_weights = []

        for band_name in self.settings['bands']:
            band = sncosmo.get_bandpass(band_name)

            band_transmission = band(10 ** (band_pad_log_wave))

            # Convolve the bands to match the sampling of the spectrum.
            band_conv_transmission = jnp.interp(band_wave, 10 ** band_pad_log_wave, band_transmission)

            band_weight = (
                    band_wave
                    * band_conv_transmission
                    / sncosmo.constants.HC_ERG_AA
                    / ref.zpbandflux(band)
                    * 10 ** (0.4 * -20.)
            )

            dlamba = jnp.diff(band_wave)
            dlamba = jnp.r_[dlamba, dlamba[-1]]

            num = band_wave * band_conv_transmission * dlamba
            denom = jnp.sum(num)
            band_weight = num / denom

            band_weights.append(band_weight)

        # Get the locations that should be sampled at redshift 0. We can scale these to
        # get the locations at any redshift.
        band_interpolate_locations = jnp.arange(
            0,
            self.settings['spectrum_bins'] * self.settings['band_oversampling'],
            self.settings['band_oversampling']
        )

        # Save the variables that we need to do interpolation.
        self.band_interpolate_locations = device_put(band_interpolate_locations)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = jnp.array(band_weights)
        self.model_wave = 10 ** (model_log_wave)

    def _calculate_band_weights(self, redshifts):
        """Calculate the band weights for a given set of redshifts

        We have precomputed the weights for each bandpass, so we simply interpolate
        those weights at the desired redshifts. We are working in log-wavelength, so a
        change in redshift just gives us a shift in indices.

        Parameters
        ----------
        redshifts : List[float]
            Redshifts to calculate the band weights at

        Returns
        -------
        `~numpy.ndarray`
            Band weights for each redshift/band combination
        """
        # Figure out the locations to sample at for each redshift.
        locs = (
                self.band_interpolate_locations
                + jnp.log10(1 + redshifts)[:, None] / self.band_interpolate_spacing
        )

        flat_locs = locs.flatten()

        # Linear interpolation
        int_locs = flat_locs.astype(jnp.int32)
        remainders = flat_locs - int_locs

        start = self.band_interpolate_weights[..., int_locs]
        end = self.band_interpolate_weights[..., int_locs + 1]

        flat_result = remainders * end + (1 - remainders) * start
        result = flat_result.reshape((-1,) + locs.shape).transpose(1, 2, 0)

        # We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        # Hack fix maybe
        sum = jnp.sum(result, axis=1)
        result /= sum[:, None, :]

        return result

    def get_flux_batch(self, theta, Av, W0, W1, eps, Ds, redshifts, band_indices):
        num_batch = theta.shape[0]
        J_t = jnp.reshape(self.J_t, (-1, *self.J_t.shape))
        J_t = jnp.repeat(J_t, num_batch, axis=0)
        J_t_hsiao = jnp.reshape(self.J_t_hsiao, (-1, *self.J_t_hsiao.shape))
        J_t_hsiao = jnp.repeat(J_t_hsiao, num_batch, axis=0)
        W0 = jnp.reshape(W0, (-1, *self.W0.shape))
        W0 = jnp.repeat(W0, num_batch, axis=0)
        W1 = jnp.reshape(W1, (-1, *self.W1.shape))
        W1 = jnp.repeat(W1, num_batch, axis=0)

        W = W0 + theta[..., None, None] * W1 + eps

        WJt = jnp.matmul(W, J_t)
        W_grid = jnp.matmul(self.J_l_T, WJt)

        HJt = jnp.matmul(self.hsiao_flux, J_t_hsiao)
        H_grid = jnp.matmul(self.J_l_T_hsiao, HJt)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)

        num_observations = band_indices.shape[0]

        band_weights = self._calculate_band_weights(redshifts)
        batch_indices = (
            jnp.arange(num_batch)
            .repeat(num_observations)
        ).astype(int)

        obs_band_weights = (
            band_weights[batch_indices, :, band_indices.T.flatten()]
            .reshape((num_batch, num_observations, -1))
            .transpose(0, 2, 1)
        )

        # Extinction----------------------------------------------------------
        Rv = 2.610
        f99_x0 = 4.596
        f99_gamma = 0.99
        f99_c2 = -0.824 + 4.717 / Rv
        f99_c1 = 2.030 - 3.007 * f99_c2
        f99_c3 = 3.23
        f99_c4 = 0.41
        f99_c5 = 5.9
        f99_d1 = self.xk[7] ** 2 / ((self.xk[7] ** 2 - f99_x0 ** 2) ** 2 + (f99_gamma * self.xk[7]) ** 2)
        f99_d2 = self.xk[8] ** 2 / ((self.xk[8] ** 2 - f99_x0 ** 2) ** 2 + (f99_gamma * self.xk[8]) ** 2)
        yk = jnp.zeros((num_batch, 9))
        yk = yk.at[:, 0].set(-Rv)
        yk = yk.at[:, 1].set(0.26469 * Rv / 3.1 - Rv)
        yk = yk.at[:, 2].set(0.82925 * Rv / 3.1 - Rv)
        yk = yk.at[:, 3].set(-0.422809 + 1.00270 * Rv + 2.13572e-4 * Rv ** 2 - Rv)
        yk = yk.at[:, 4].set(-5.13540e-2 + 1.00216 * Rv - 7.35778e-5 * Rv ** 2 - Rv)
        yk = yk.at[:, 5].set(0.700127 + 1.00184 * Rv - 3.32598e-5 * Rv ** 2 - Rv)
        yk = yk.at[:, 6].set(1.19456 + 1.01707 * Rv - 5.46959e-3 * Rv ** 2 + 7.97809e-4 * Rv ** 3 - 4.45636e-5 * Rv ** 4 - Rv)
        yk = yk.at[:, 7].set(f99_c1 + f99_c2 * self.xk[7] + f99_c3 * f99_d1)
        yk = yk.at[:, 8].set(f99_c1 + f99_c2 * self.xk[8] + f99_c3 * f99_d2)

        A = Av[..., None] * (1 + (self.M_fitz_block @ yk.T).T / Rv) #Rv[..., None]
        f_A = 10 ** (-0.4 * A)
        model_spectra = model_spectra * f_A[..., None]

        model_flux = jnp.sum(model_spectra * obs_band_weights, axis=1).T

        model_flux = model_flux * 10 ** (-0.4 * (self.M0 + Ds))

        return model_flux

    def fit_model(self, obs):
        sample_size = self.data.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        # for sn_index in numpyro.plate('SNe', sample_size):
        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            # Rv = numpyro.sample(f'RV', dist.Normal(2.610, 0.001))
            Av = numpyro.sample(f'AV', dist.Exponential(1 / 0.194))
            eps_mu = jnp.zeros(N_knots_sig)
            eps = numpyro.sample(f'eps', dist.MultivariateNormal(eps_mu, scale_tril=self.L_Sigma))
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            band_indices = obs[-3, :, sn_index].astype(int).T
            redshift = obs[-2, 0, sn_index]
            flux = self.get_flux_batch(theta, Av, self.W0, self.W1, eps, redshift, band_indices)
            numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def fit(self, dataset):
        self.process_dataset(dataset)
        rng = PRNGKey(123)
        # numpyro.render_model(self.fit_model, model_args=(self.data,), filename='fit_model.pdf')
        nuts_kernel = NUTS(self.fit_model, adapt_step_size=True, init_strategy=init_to_median())
        mcmc = MCMC(nuts_kernel, num_samples=250, num_warmup=250, num_chains=1)
        mcmc.run(rng, self.data)  # self.rng,
        return mcmc

    def fit_assess(self, params, yaml_dir):
        with open(os.path.join('results', f'{yaml_dir}.yaml'), 'r') as file:
            result = yaml.load(file, yaml.Loader)
        # Theta
        params['fit_theta_mu'] = np.median(result['theta'], axis=0)
        params['fit_theta_std'] = np.std(result['theta'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        ax[0].errorbar(params.theta, params.fit_theta_mu, yerr=params.fit_theta_std, fmt='x')
        ax[1].errorbar(params.theta, params.fit_theta_mu + params.theta, yerr=params.fit_theta_std, fmt='x')
        plt.show()
        print(params[['theta', 'fit_theta_mu', 'fit_theta_std']])
        # Av
        params['fit_AV_mu'] = np.median(result['AV'], axis=0)
        params['fit_AV_std'] = np.std(result['AV'], axis=0)
        plt.errorbar(params.AV, params.fit_AV_mu, yerr=params.fit_AV_std, fmt='x')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        print(params[['AV', 'fit_AV_mu', 'fit_AV_std']])

    def train_model(self, obs):
        sample_size = self.data.shape[-1]
        N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        W_mu = jnp.zeros(N_knots)
        W0 = numpyro.sample('W0', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W1 = numpyro.sample('W1', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W0 = jnp.reshape(W0, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = jnp.reshape(W1, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        sigmaepsilon = numpyro.sample('sigmaepsilon', dist.HalfNormal(0.25 * jnp.ones(N_knots_sig)))
        L_Omega = numpyro.sample('L_Omega', dist.LKJCholesky(N_knots_sig))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon), L_Omega)
        sigma0 = numpyro.sample('sigma0', dist.HalfCauchy(0.1))

        # for sn_index in pyro.plate('SNe', sample_size):
        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            # Rv = numpyro.sample(f'RV', dist.Normal(2.610, 0.001))
            Av = numpyro.sample(f'AV', dist.Exponential(1 / 0.194))
            eps_mu = jnp.zeros(N_knots_sig)
            eps = numpyro.sample('eps', dist.MultivariateNormal(eps_mu, scale_tril=L_Sigma))
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            band_indices = obs[-3, :, sn_index].astype(int).T
            redshift = obs[-2, 0, sn_index]
            muhat = obs[-1, 0, sn_index]
            muhat_err = 5 / (redshift * jnp.log(10)) * self.sigma_pec
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            start = time.time()
            end = time.time()
            elapsed = end - start
            numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T) # _{sn_index}

    def train(self, dataset):
        self.process_dataset(dataset)
        self.integ_time = 0
        self.total = 0
        self.count = 0
        self.thetas = []
        rng = PRNGKey(123)
        # numpyro.render_model(self.train_model, model_args=(self.data,), filename='train_model.pdf')
        nuts_kernel = NUTS(self.train_model, adapt_step_size=True, init_strategy=init_to_median())
        mcmc = MCMC(nuts_kernel, num_samples=250, num_warmup=250, num_chains=1)
        mcmc.run(rng, self.data)  # self.rng,
        print(f'{self.total * self.data.shape[1]} flux integrals for {self.total} objects in {self.integ_time} seconds')
        print(f'Average per object: {self.integ_time / self.total}')
        print(f'Average per integral: {self.integ_time / (self.total * self.data.shape[1])}')
        print(np.array(self.thetas))
        return mcmc

    def train_assess(self, params, yaml_dir):
        with open(os.path.join('results', f'{yaml_dir}.yaml'), 'r') as file:
            result = yaml.load(file, yaml.Loader)
        # Theta
        params['fit_theta_mu'] = np.median(result['theta'], axis=0)
        params['fit_theta_std'] = np.std(result['theta'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.theta, params.fit_theta_mu, yerr=params.fit_theta_std, fmt='x')
        ax[1].errorbar(params.theta, params.fit_theta_mu - params.theta, yerr=params.fit_theta_std, fmt='x')
        ax[1].set_xlabel(rf'True $\theta$')
        ax[0].set_ylabel(rf'Fit $\theta$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        print(params[['theta', 'fit_theta_mu', 'fit_theta_std']])
        # Av
        params['fit_AV_mu'] = np.median(result['AV'], axis=0)
        params['fit_AV_std'] = np.std(result['AV'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].scatter(params.AV, params.fit_AV_mu) #, yerr=params.fit_AV_std, fmt='x')
        ax[1].scatter(params.AV, params.fit_AV_mu - params.AV)#, yerr=params.fit_AV_std, fmt='x')
        ax[1].set_xscale('log')
        ax[0].set_yscale('log')
        # ax[1].set_yscale('log')
        ax[1].set_xlabel(rf'True $A_V$')
        ax[0].set_ylabel(rf'Fit $A_V$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    def process_dataset(self, dataset):
        all_data = []
        self.t = None
        for lc_ind, lc in enumerate(dataset.light_curves):
            lc = lc.to_pandas()
            lc = lc.astype({'band': str})
            lc[['flux', 'fluxerr']] = lc[['flux', 'fluxerr']] # / self.scale
            lc['band'] = lc['band'].apply(lambda band: band[band.find("'") + 1: band.rfind("'")])
            lc['band'] = lc['band'].apply(lambda band: self.band_dict[band])
            lc['redshift'] = dataset.meta[lc_ind][-1]
            lc['dist_mod'] = self.cosmo.distmod(dataset.meta[lc_ind][-1])
            lc = lc.sort_values('time')
            if self.t is None:
                self.t = lc.time.values
            lc = lc.values
            lc = np.reshape(lc, (-1, *lc.shape))
            all_data.append(lc)
        all_data = np.concatenate(all_data)
        all_data = np.swapaxes(all_data, 0, 2)
        self.data = device_put(all_data)
        self.J_t = device_put(spline_utils.spline_coeffs_irr(self.t, self.tau_knots, self.KD_t).T)
        self.J_t_hsiao = device_put(spline_utils.spline_coeffs_irr(self.t, self.hsiao_t,
                                                                              self.KD_t_hsiao).T)

    def fit_from_results(self, dataset, input_file):
        self.process_dataset(dataset)
        with open(os.path.join('results', f'{input_file}.yaml'), 'r') as file:
            result = yaml.load(file, yaml.Loader)
        N = result['theta'].shape[1]
        W0 = np.reshape(np.mean(result['W0'], axis=0), (6, 6), order='F')
        W1 = np.reshape(np.mean(result['W1'], axis=0), (6, 6), order='F')
        eps = np.reshape(np.mean(result['eps'], axis=0), (N, 4, 6), order='F')
        new_eps = np.zeros((eps.shape[0], eps.shape[1] + 2, eps.shape[2]))
        theta, Av = np.mean(result['theta'], axis=0), np.mean(result['AV'], axis=0)
        new_eps[:, 1:-1, :] = eps
        eps = new_eps
        band_indices = self.data[-2, :, :].astype(int)
        redshift = self.data[-1, 0, :]
        model_flux = self.get_flux_batch(theta, Av, W0, W1, eps, redshift, band_indices)
        for _ in range(10):
            plt.figure()
            for i in range(4):
                inds = band_indices[:, 0] == i
                plt.errorbar(self.t[inds], self.data[1, inds, _], yerr=self.data[2, inds, _], fmt='x')
                plt.scatter(self.t[inds], model_flux[inds, _])
        plt.show()

    def test_params(self, dataset, params):
        W0 = np.array([[-0.11843238,  0.5618372 ,  0.38466904,  0.23944603,
              -0.48216674,  0.63915277],
             [ 0.08652873,  0.18803605,  0.26729473,  0.31465054,
               0.39766428,  0.1856934 ],
             [ 0.09866332,  0.24971685,  0.31222486,  0.27499378,
               0.29981762,  0.2305843 ],
             [ 0.1864955 ,  0.3214951 ,  0.34273627,  0.29547283,
               0.43862557,  0.29078126],
             [ 0.29226252,  0.39753425,  0.36405647,  0.47865516,
               0.44378856,  0.43702376],
             [ 1.1537213 ,  1.0743428 ,  0.0494406 ,  0.46162465,
               1.333291  ,  0.04616207]])
        W1 = np.array([[ 0.21677628, -0.15750809,  0.7421827 ,  0.0212789 ,
               0.64179885, -0.3533681 ],
             [ 0.41216654,  0.20722015,  0.2119322 ,  0.4584896 ,
               0.39576882,  0.3574507 ],
             [ 0.29122993,  0.13642277,  0.20810808,  0.11366853,
               0.4389719 ,  0.23162062],
             [ 0.29141757,  0.0731663 ,  0.12748136, -0.01537234,
               0.33767635,  0.31357116],
             [ 0.21928684,  0.18876176,  0.12241068,  0.08746054,
               0.36593395,  0.4919144 ],
             [ 0.08234484,  0.21387804, -0.3760478 ,  1.0113571 ,
               1.0101043 ,  1.4508004 ]])
        self.process_dataset(dataset)
        band_indices = self.data[-2, :, 0:1].astype(int)
        redshift = self.data[-1, 0, 0:1]
        theta, Av = params.theta.values, params.AV.values
        flux = self.get_flux_batch(theta, Av, W0, W1, redshift, band_indices)
        for i in range(4):
            inds = band_indices[:, 0] == i
            plt.scatter(self.t[inds], flux[inds, 0])
            plt.errorbar(self.t[inds], self.data[1, inds, 0], yerr=self.data[2, inds, 0], fmt='x')
        plt.show()

    def save_results_to_yaml(self, result, output_path):
        results_dict = {}
        result = result.get_samples()
        fit_params = result.keys()
        if 'W0' not in fit_params:
            results_dict['W0'] = self.W0
            results_dict['W1'] = self.W1
            results_dict['L_sigma'] = self.L_Sigma
        for k, v in result.items():
            results_dict[k] = v
        with open(os.path.join('results', f'{output_path}.yaml'), 'w') as file:
            yaml.dump(result, file, default_flow_style=False)


# -------------------------------------------------

def get_band_effective_wavelength(band):
    """Calculate the effective wavelength of a band

    The results of this calculation are cached, and the effective wavelength will only
    be calculated once for each band.

    Parameters
    ----------
    band : str
        Name of a band in the `sncosmo` band registry

    Returns
    -------
    float
        Effective wavelength of the band.
    """
    return sncosmo.get_bandpass(band).wave_eff


if __name__ == '__main__':
    dataset_path = 'data/bayesn_sim_team_z0.1_25000.h5'
    dataset = lcdata.read_hdf5(dataset_path)[:1000]
    bands = set()
    for lc in dataset.light_curves:
        bands = bands.union(lc['band'])
    bands = np.array(sorted(bands, key=get_band_effective_wavelength))

    param_path = 'data/bayesn_sim_team_z0.1_25000_params.csv'
    params = pd.read_csv(param_path)

    pd_dataset = dataset.meta.to_pandas()
    pd_dataset = pd_dataset.astype({'object_id': int})
    params = pd_dataset.merge(params, on='object_id')

    model = Model(bands, device='cuda')
    # result = model.fit(dataset)
    result = model.train(dataset)
    # inf_data = az.from_numpyro(result)
    # print(az.summary(inf_data))
    # model.save_results_to_yaml(result, 'gpu_train_dist')
    # model.fit_assess(params, 'fit_test')
    model.fit_from_results(dataset, 'gpu_train')
    # model.train_assess(params, 'gpu_train_Av')




