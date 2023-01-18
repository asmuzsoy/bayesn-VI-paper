import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, HMC, init_to_median, init_to_sample, init_to_value
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
import arviz

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})

# jax.config.update('jax_platform_name', 'cpu')
# numpyro.set_host_device_count(4)

print(jax.devices())


class Model(object):
    def __init__(self, ignore_unknown_settings=False, settings={},
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}):
        self.data = None
        bands = ['ps1::g', 'ps1::r', 'ps1::i', 'ps1::z']
        self.band_dict = {band[-1]: i for i, band in enumerate(bands)}
        self.settings = parse_settings(bands, settings,
                                       ignore_unknown_settings=ignore_unknown_settings)
        self.M0 = device_put(jnp.array(-19.5))
        self.scale = 1e18
        self.device_scale = device_put(jnp.array(self.scale))
        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.sigma_pec = device_put(jnp.array(150 / 3e5))

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
        self.RV_MW = 3.1

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

    def _calculate_band_weights(self, redshifts, ebv):
        print(redshifts)
        print(ebv)
        raise ValueError('Nope')
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

        """# We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        # Apply MW extinction
        abv = self.RV_MW * ebv
        mw_array = jnp.zeros((result.shape[0], result.shape[1]))
        for i, val in enumerate(abv):
            mw = jnp.power(10, -0.4 * extinction.fitzpatrick99(self.model_wave, val, self.RV_MW))
            mw_array = mw_array.at[i, :].set(mw)

        result = result * mw_array[..., None]

        # Hack fix maybe
        sum = jnp.sum(result, axis=1)
        result /= sum[:, None, :]"""

        # Apply MW extinction
        abv = self.RV_MW * ebv
        mw_array = jnp.zeros((result.shape[0], result.shape[1]))
        for i, val in enumerate(abv):
            mw = jnp.power(10, -0.4 * extinction.fitzpatrick99(self.model_wave, val, self.RV_MW))
            mw_array = mw_array.at[i, :].set(mw)

        result = result * mw_array[..., None]

        # Hack fix maybe
        sum = jnp.sum(result, axis=1)
        result /= sum[:, None, :]

        # We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        return result

    def get_flux_batch(self, theta, Av, W0, W1, eps, Ds, Rv, redshifts, ebv, band_indices, flag):
        num_batch = theta.shape[0]
        W0 = jnp.reshape(W0, (-1, *self.W0.shape))
        W0 = jnp.repeat(W0, num_batch, axis=0)
        W1 = jnp.reshape(W1, (-1, *self.W1.shape))
        W1 = jnp.repeat(W1, num_batch, axis=0)

        W = W0 + theta[..., None, None] * W1 + eps

        WJt = jnp.matmul(W, self.J_t)
        W_grid = jnp.matmul(self.J_l_T, WJt)

        HJt = jnp.matmul(self.hsiao_flux, self.J_t_hsiao)
        H_grid = jnp.matmul(self.J_l_T_hsiao, HJt)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)

        num_observations = band_indices.shape[0]

        # band_weights = self._calculate_band_weights(redshifts, ebv)
        batch_indices = (
            jnp.arange(num_batch)
            .repeat(num_observations)
        ).astype(int)

        obs_band_weights = (
            self.band_weights[batch_indices, :, band_indices.T.flatten()]
            .reshape((num_batch, num_observations, -1))
            .transpose(0, 2, 1)
        )

        # Extinction----------------------------------------------------------
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
        yk = yk.at[:, 6].set(
            1.19456 + 1.01707 * Rv - 5.46959e-3 * Rv ** 2 + 7.97809e-4 * Rv ** 3 - 4.45636e-5 * Rv ** 4 - Rv)
        yk = yk.at[:, 7].set(f99_c1 + f99_c2 * self.xk[7] + f99_c3 * f99_d1)
        yk = yk.at[:, 8].set(f99_c1 + f99_c2 * self.xk[8] + f99_c3 * f99_d2)

        A = Av[..., None] * (1 + (self.M_fitz_block @ yk.T).T / Rv)  # Rv[..., None]
        f_A = 10 ** (-0.4 * A)
        model_spectra = model_spectra * f_A[..., None]

        model_flux = jnp.sum(model_spectra * obs_band_weights, axis=1).T
        model_flux = model_flux * 10 ** (-0.4 * (self.M0 + Ds))
        model_flux *= self.device_scale
        model_flux *= flag

        return model_flux

    def fit_model(self, obs):
        sample_size = self.data.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        sigma0 = 0.103

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
            muhat = obs[-1, 0, sn_index]
            muhat_err = 5 / (redshift * jnp.log(10)) * self.sigma_pec
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(theta, Av, self.W0, self.W1, eps, Ds, redshift, band_indices)
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
        print(result['theta'].shape)
        return
        # Add dist mods
        params['distmod'] = self.cosmo.distmod(params.redshift.values).value
        # Theta
        params['fit_theta_mu'] = np.median(result['theta'], axis=0)
        params['fit_theta_std'] = np.std(result['theta'], axis=0)
        if (params.fit_theta_mu - params.theta).min() < -4:
            sign = -1
        else:
            sign = 1
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.theta, sign * params.fit_theta_mu, yerr=params.fit_theta_std, fmt='x')
        ax[1].errorbar(params.theta, sign * params.fit_theta_mu - params.theta, yerr=params.fit_theta_std, fmt='x')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
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
        ax[0].scatter(params.AV, params.fit_AV_mu)  # , yerr=params.fit_AV_std, fmt='x')
        ax[1].scatter(params.AV, params.fit_AV_mu - params.AV)  # , yerr=params.fit_AV_std, fmt='x')
        ax[1].set_xscale('log')
        ax[0].set_yscale('log')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
        # ax[1].set_yscale('log')
        ax[1].set_xlabel(rf'True $A_V$')
        ax[0].set_ylabel(rf'Fit $A_V$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        # dist_mod
        muhat = params['distmod'].values
        muhat_err = 5 / (params.redshift.values * jnp.log(10)) * self.sigma_pec
        sigma0 = 0.103
        print(params.keys())
        mu = (result['Ds'] * np.power(muhat_err, 2) + muhat * np.power(sigma0, 2)) \
             / (muhat_err * muhat_err + sigma0 * sigma0)
        std = np.sqrt(np.power(sigma0 * muhat_err, 2) / (muhat_err * muhat_err + sigma0 * sigma0))
        distmod = np.random.normal(mu, std)
        del_M = result['Ds'] - distmod
        result['distmod'] = distmod
        result['del_M'] = del_M
        params['fit_distmod_mu'] = np.median(result['distmod'], axis=0)
        params['fit_distmod_std'] = np.std(result['distmod'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.distmod, params.fit_distmod_mu, yerr=params.fit_distmod_std, fmt='x')
        ax[1].errorbar(params.distmod, params.fit_distmod_mu - params.distmod, yerr=params.fit_distmod_std, fmt='x')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
        ax[1].set_xlabel(rf'True $\mu$')
        ax[0].set_ylabel(rf'Fit $\mu$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        # dist_mod
        params['fit_delM_mu'] = np.median(result['del_M'], axis=0)
        params['fit_delM_std'] = np.std(result['del_M'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.del_M, params.fit_delM_mu, yerr=params.fit_delM_std, fmt='x')
        ax[1].errorbar(params.del_M, params.fit_delM_mu - params.del_M, yerr=params.fit_delM_std, fmt='x')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
        ax[1].set_xlabel(rf'True $\delta M$')
        ax[0].set_ylabel(rf'Fit $\delta M$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        # epsilon
        correlation_array = np.zeros((24, 24))
        for i in range(24):
            params[f'fit_epsilon{i}_mu'] = np.median(result['eps'][..., i], axis=0)
            params[f'fit_epsilon{i}_std'] = np.std(result['eps'][..., i], axis=0)
            for j in range(24):
                correlation_array[i, j] = pearsonr(params[f'epsilon_{j}'], params[f'fit_epsilon{i}_mu'])[0]
        for row in correlation_array:
            print(np.max(row))

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
        # sigma0 = numpyro.sample('sigma0', dist.HalfCauchy(0.1))
        sigma0_tform = numpyro.sample('sigma0_tform', dist.Uniform(0, jnp.pi / 2))
        sigma0 = numpyro.deterministic('sigma0', 1.0 * jnp.tan(sigma0_tform))
        Rv = numpyro.sample('Rv', dist.Uniform(1, 5))
        # tauA = numpyro.sample('tauA', dist.HalfCauchy())
        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))

        # for sn_index in pyro.plate('SNe', sample_size):
        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            Av = numpyro.sample(f'AV', dist.Exponential(1 / tauA))
            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps = numpyro.deterministic('eps', jnp.matmul(eps_tform, L_Sigma))
            # eps = numpyro.sample('eps', dist.MultivariateNormal(eps_mu, scale_tril=L_Sigma))
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]
            flag = obs[-1, :, sn_index].T
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, Rv, redshift, ebv, band_indices, flag)
            """for i in range(4):
                inds = band_indices[:, 0] == i
                plt.scatter(self.t[inds], flux[inds, 0])
                plt.errorbar(self.t[inds], obs[1, inds, 0], yerr=obs[2, inds, 0], fmt='x')
            plt.show()
            for i in range(4):
                inds = band_indices[:, 1] == i
                plt.scatter(self.t[inds], flux[inds, 1])
                plt.errorbar(self.t[inds], obs[1, inds, 1], yerr=obs[2, inds, 1], fmt='x')
            plt.show()"""
            numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)  # _{sn_index}

    def initial_guess(self, n_chains=1, reference_model="M20", RV_init=3.0, tauA_init=0.3):
        # Set hyperparameter initialisations
        # Set initialisations (these will be jittered chain by chain)
        param_root = 'model_files/T21_model'
        W0_init = np.loadtxt(f'{param_root}/W0.txt')
        n_lknots, n_tauknots = W0_init.shape
        W0_init = W0_init.flatten(order='F')
        W1_init = np.loadtxt(f'{param_root}/W1.txt').flatten(order='F')
        RV_init, tauA_init = np.loadtxt(f'{param_root}/M0_sigma0_RV_tauA.txt')[[2, 3]]

        # I should remove all of this hardcoding
        n_eps = (n_lknots - 2) * n_tauknots
        sigma0_init = 0.1
        sigmaepsilon_init = 0.1 * np.ones(n_eps)
        L_Omega_init = np.eye(n_eps)

        n_sne = self.data.shape[-1]

        # Prepare initial guesses
        # I should make the jittering more free for the user to control
        param_init = {
            'W0': np.zeros((n_chains, *W0_init.shape)),
            'W1': np.zeros((n_chains, *W1_init.shape)),
            'Rv': np.zeros(n_chains),
            'tauA': np.zeros(n_chains),
            'sigma0': np.zeros(n_chains),
            'sigmaepsilon': np.zeros((n_chains, *sigmaepsilon_init.shape)),
            'L_Omega': np.zeros((n_chains, *L_Omega_init.shape)),
            'Av': np.zeros((n_chains, n_sne)),
            'theta': np.zeros((n_chains, n_sne)),
            'epsilon': np.zeros((n_chains, n_sne, n_eps)),
            'Ds': np.zeros((n_chains, n_sne))
        }
        param_init = {}
        tauA_ = tauA_init + np.random.normal(0, 0.01)
        while tauA_ < 0:
            tauA_ = tauA_init + np.random.normal(0, 0.01)
        sigma0_ = sigma0_init + np.random.normal(0, 0.01)
        param_init['W0'] = W0_init + np.random.normal(0, 0.01, W0_init.shape[0])
        param_init['W1'] = W1_init + np.random.normal(0, 0.01, W1_init.shape[0])
        param_init['Rv'] = RV_init  # + np.random.uniform(1.0 - RV_init, 5.0 - RV_init)
        param_init['tauA'] = tauA_
        param_init['sigma0'] = sigma0_
        param_init['sigmaepsilon'] = sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape)
        param_init['L_Omega'] = L_Omega_init
        param_init['theta'] = np.random.normal(0, 1, n_sne)
        param_init['Av'] = np.random.exponential(tauA_, n_sne)
        param_init['epsilon'] = np.random.normal(0, 1, (n_sne, n_eps))
        param_init['Ds'] = np.random.normal(self.data[-3, 0, :], sigma0_)

        return param_init

    def train(self, num_samples, num_warmup, num_chains, output, chain_method='parallel', init_strategy='median'):
        self.process_dataset(mode='training')
        if init_strategy == 'value':
            init_strategy = init_to_value(values=self.initial_guess())
        elif init_strategy == 'median':
            init_strategy = init_to_median()
        elif init_strategy == 'sample':
            init_strategy = init_to_sample()
        else:
            raise ValueError('Invalid init strategy, must be one of value, median and sample')
        self.band_weights = self._calculate_band_weights(self.data[-4, 0, :], self.data[-2, 0, :])
        # rng = jnp.array([PRNGKey(12), PRNGKey(34), PRNGKey(56), PRNGKey(78)])
        rng = PRNGKey(10)
        # rng, rng_ = PRNGKey(10)
        # numpyro.render_model(self.train_model, model_args=(self.data,), filename='train_model.pdf')
        nuts_kernel = NUTS(self.train_model, adapt_step_size=True, target_accept_prob=0.8, init_strategy=init_strategy)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains,
                    chain_method=chain_method)
        mcmc.run(rng, self.data)
        mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        self.train_postprocess(samples, output)

    def train_postprocess(self, samples, output):
        if not os.path.exists(os.path.join('results', output)):
            os.mkdir(os.path.join('results', output))
        with open(os.path.join('results', output, 'initial_chains.pkl'), 'wb') as file:
            pickle.dump(samples, file)
        # Sign flipping-----------------
        J_R = spline_utils.spline_coeffs_irr([6200.0], self.l_knots, spline_utils.invKD_irr(self.l_knots))
        J_10 = spline_utils.spline_coeffs_irr([10.0], self.tau_knots, spline_utils.invKD_irr(self.tau_knots))
        J_0 = spline_utils.spline_coeffs_irr([0.0], self.tau_knots, spline_utils.invKD_irr(self.tau_knots))
        W1 = np.reshape(samples['W1'], (
        samples['W1'].shape[0], samples['W1'].shape[1], self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        N_chains = W1.shape[0]
        sign = np.zeros(N_chains)
        for chain in range(N_chains):
            chain_W1 = np.mean(W1[chain, ...], axis=0)
            chain_sign = np.sign(
                np.squeeze(np.matmul(J_R, np.matmul(chain_W1, J_10.T))) - np.squeeze(
                    np.matmul(J_R, np.matmul(chain_W1, J_0.T))))
            sign[chain] = chain_sign
        samples["W1"] = samples["W1"] * sign[:, None, None]
        samples["theta"] = samples["theta"] * sign[:, None, None]
        # Modify W1 and theta----------------
        theta_std = np.std(samples["theta"], axis=2)
        samples['theta'] = samples['theta'] / theta_std[..., None]
        samples['W1'] = samples['W1'] * theta_std[..., None]
        # Save convergence data for each parameter to csv file
        summary = arviz.summary(samples)
        summary.to_csv(os.path.join('results', output, 'fit_summary.csv'))
        with open(os.path.join('results', output, 'chains.pkl'), 'wb') as file:
            pickle.dump(samples, file)
        # Save best fit global params to files for easy inspection and reading in------
        W0 = np.mean(samples['W0'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = np.mean(samples['W1'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]),
                                                         order='F')
        sigmaepsilon = np.mean(samples['sigmaepsilon'], axis=[0, 1])
        L_Omega = np.mean(samples['L_Omega'], axis=[0, 1])
        L_Sigma = np.matmul(np.diag(np.mean(samples['sigmaepsilon'], axis=[0, 1])),
                            np.mean(samples['L_Omega'], axis=[0, 1]))
        sigma0 = np.mean(samples['sigma0'])
        Rv = np.mean(samples['Rv'])
        tauA = np.mean(samples['tauA'])
        M0_sigma0_RV_tauA = np.array([self.M0, sigma0, Rv, tauA])
        np.savetxt(os.path.join('results', output, 'W0.txt'), W0, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'W1.txt'), W1, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'sigmaepsilon.txt'), sigmaepsilon, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'L_Omega.txt'), L_Omega, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'L_Sigma.txt'), L_Sigma, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'M0_sigma0_RV_tauA.txt'), M0_sigma0_RV_tauA, delimiter="\t",
                   fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'l_knots.txt'), self.l_knots, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'tau_knots.txt'), self.tau_knots, delimiter="\t", fmt="%.3f")

        """global_param_dict = {
            'W0': repr(np.round(np.mean(samples['W0'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]), order='F'), 3).tolist()),
            'W1': repr(np.mean(samples['W1'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]),
                                                              order='F').tolist()),
            'sigmaepsilon': repr(np.mean(samples['sigmaepsilon'], axis=[0, 1])),
            'L_Omega': repr(np.mean(samples['L_Omega'], axis=[0, 1])),
            'L_Sigma': repr(np.matmul(np.diag(np.mean(samples['sigmaepsilon'], axis=[0, 1])), np.mean(samples['L_Omega'], axis=[0, 1]))),
            'sigma0': np.mean(samples['sigma0'])
        }"""

    def train_assess(self, params, yaml_dir):
        with open(os.path.join('results', f'{yaml_dir}.yaml'), 'r') as file:
            result = yaml.load(file, yaml.Loader)
        print(result['theta'].shape)
        print(result.keys())
        raise ValueError('Nope')
        # Add dist mods
        params['distmod'] = self.cosmo.distmod(params.redshift.values).value
        # Theta
        params['fit_theta_mu'] = np.median(result['theta'], axis=0)
        params['fit_theta_std'] = np.std(result['theta'], axis=0)
        if (params.fit_theta_mu - params.theta).min() < -4:
            sign = -1
        else:
            sign = 1
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.theta, sign * params.fit_theta_mu, yerr=params.fit_theta_std, fmt='x')
        ax[1].errorbar(params.theta, sign * params.fit_theta_mu - params.theta, yerr=params.fit_theta_std, fmt='x')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
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
        ax[0].scatter(params.AV, params.fit_AV_mu)  # , yerr=params.fit_AV_std, fmt='x')
        ax[1].scatter(params.AV, params.fit_AV_mu - params.AV)  # , yerr=params.fit_AV_std, fmt='x')
        ax[1].set_xscale('log')
        ax[0].set_yscale('log')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
        # ax[1].set_yscale('log')
        ax[1].set_xlabel(rf'True $A_V$')
        ax[0].set_ylabel(rf'Fit $A_V$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        # dist_mod
        muhat = params['distmod'].values
        muhat_err = 5 / (params.redshift.values * jnp.log(10)) * self.sigma_pec
        sigma0 = np.mean(result['sigma0'], axis=0)
        mu = (result['Ds'] * np.power(muhat_err, 2) + muhat * np.power(result['sigma0'][..., None], 2)) \
             / (muhat_err * muhat_err + result['sigma0'][..., None] * result['sigma0'][..., None])
        std = np.sqrt(np.power(sigma0 * muhat_err, 2) / (muhat_err * muhat_err + sigma0 * sigma0))
        distmod = np.random.normal(mu, std)
        del_M = result['Ds'] - distmod
        result['distmod'] = distmod
        result['del_M'] = del_M
        params['fit_distmod_mu'] = np.median(result['distmod'], axis=0)
        params['fit_distmod_std'] = np.std(result['distmod'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.distmod, params.fit_distmod_mu, yerr=params.fit_distmod_std, fmt='x')
        ax[1].errorbar(params.distmod, params.fit_distmod_mu - params.distmod, yerr=params.fit_distmod_std, fmt='x')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
        ax[1].set_xlabel(rf'True $\mu$')
        ax[0].set_ylabel(rf'Fit $\mu$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        # dist_mod
        params['fit_delM_mu'] = np.median(result['del_M'], axis=0)
        params['fit_delM_std'] = np.std(result['del_M'], axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(9, 12), sharex='col')
        ax[0].errorbar(params.del_M, params.fit_delM_mu, yerr=params.fit_delM_std, fmt='x')
        ax[1].errorbar(params.del_M, params.fit_delM_mu - params.del_M, yerr=params.fit_delM_std, fmt='x')
        xlim = ax[0].get_xlim()
        ax[0].autoscale(tight=True)
        ax[0].plot(xlim, xlim, 'k--')
        ax[1].hlines(0, *xlim, colors='k', ls='--')
        ax[1].set_xlabel(rf'True $\delta M$')
        ax[0].set_ylabel(rf'Fit $\delta M$')
        ax[1].set_ylabel(rf'Residual')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    def process_dataset(self, mode='training'):
        dataset_path = 'data/bayesn_sim_team_z0.1_25000.h5'
        dataset = lcdata.read_hdf5(dataset_path)[:157]
        param_path = 'data/bayesn_sim_team_z0.1_25000_params.csv'
        params = pd.read_csv(param_path)
        pd_dataset = dataset.meta.to_pandas()
        pd_dataset = pd_dataset.astype({'object_id': int})
        params = pd_dataset.merge(params, on='object_id')
        all_data = []
        self.t = None
        for lc_ind, lc in enumerate(dataset.light_curves):
            lc = lc.to_pandas()
            lc = lc.astype({'band': str})
            lc['t'] = lc['time']
            lc[['flux', 'flux_err']] = lc[['flux', 'fluxerr']]  * self.scale
            lc['band'] = lc['band'].apply(lambda band: band[band.find("'") + 1: band.rfind("'")])
            lc['band'] = lc['band'].apply(lambda band: self.band_dict[band])
            lc['redshift'] = dataset.meta[lc_ind][-1]
            lc['dist_mod'] = self.cosmo.distmod(dataset.meta[lc_ind][-1])
            lc['band_indices'] = lc.band
            lc['redshift_error'] = 1e-5
            lc['MWEBV'] = 0
            lc['flag'] = 1
            lc = lc[['t', 'flux', 'flux_err', 'band_indices', 'redshift', 'redshift_error', 'dist_mod', 'MWEBV', 'flag']]
            lc = lc.sort_values('t')
            if self.t is None:
                self.t = lc.t.values
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
        theta, Av, Ds = np.mean(result['theta'], axis=0), np.mean(result['AV'], axis=0), np.mean(result['Ds'], axis=0)
        new_eps[:, 1:-1, :] = eps
        eps = new_eps
        band_indices = self.data[-3, :, :].astype(int)
        redshift = self.data[-2, 0, :]
        model_flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, redshift, band_indices)
        for _ in range(10):
            plt.figure()
            for i in range(4):
                inds = band_indices[:, 0] == i
                plt.errorbar(self.t[inds], self.data[1, inds, _], yerr=self.data[2, inds, _], fmt='x')
                plt.scatter(self.t[inds], model_flux[inds, _])
        plt.show()

    def test_params(self, dataset, params):
        W0 = np.array([[-0.11843238, 0.5618372, 0.38466904, 0.23944603,
                        -0.48216674, 0.63915277],
                       [0.08652873, 0.18803605, 0.26729473, 0.31465054,
                        0.39766428, 0.1856934],
                       [0.09866332, 0.24971685, 0.31222486, 0.27499378,
                        0.29981762, 0.2305843],
                       [0.1864955, 0.3214951, 0.34273627, 0.29547283,
                        0.43862557, 0.29078126],
                       [0.29226252, 0.39753425, 0.36405647, 0.47865516,
                        0.44378856, 0.43702376],
                       [1.1537213, 1.0743428, 0.0494406, 0.46162465,
                        1.333291, 0.04616207]])
        W1 = np.array([[0.21677628, -0.15750809, 0.7421827, 0.0212789,
                        0.64179885, -0.3533681],
                       [0.41216654, 0.20722015, 0.2119322, 0.4584896,
                        0.39576882, 0.3574507],
                       [0.29122993, 0.13642277, 0.20810808, 0.11366853,
                        0.4389719, 0.23162062],
                       [0.29141757, 0.0731663, 0.12748136, -0.01537234,
                        0.33767635, 0.31357116],
                       [0.21928684, 0.18876176, 0.12241068, 0.08746054,
                        0.36593395, 0.4919144],
                       [0.08234484, 0.21387804, -0.3760478, 1.0113571,
                        1.0101043, 1.4508004]])
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
        #with open(os.path.join('results', f'{output_path}.yaml'), 'w') as file:
        #    yaml.dump(result, file, default_flow_style=False)
        with open(os.path.join('results', f'{output_path}.pkl'), 'wb') as file:
            pickle.dump(result, file)


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
    model = Model()
    # model.fit(250, 250, 4, 'foundation_fit_4chain', 'foundation_train_Rv')
    model.train(5, 5, 4, 'simulation_train_5', chain_method='sequential', init_strategy='median')
    # model.train_postprocess()
    # result.print_summary()
    # model.save_results_to_yaml(result, 'foundation_train_4chain')
    # model.fit_assess(params, '4chain_fit_test')
    # model.fit_from_results('foundation_train')
    # model.train_assess(params, 'gpu_train_dist')
