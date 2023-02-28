import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, HMC, init_to_median, init_to_sample, init_to_value
import numpyro.distributions as dist
import h5py
import sncosmo
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
import astropy.constants as const
import matplotlib as mpl
from matplotlib import rc
import arviz
import extinction
import timeit
from plotting_utils import corner
from astropy.io import fits

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})
# mpl.use('macosx')

#jax.config.update('jax_platform_name', 'cpu')


class Model(object):
    def __init__(self, num_devices=4, enable_x64=False, load_model='T21_model',
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}, obsmodel_file='data/SNmodel_pb_obsmode_map.txt'):
        # Settings for jax/numpyro
        numpyro.set_host_device_count(num_devices)
        jax.config.update('jax_enable_x64', enable_x64)
        print('Current devices:', jax.devices())

        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.data = None
        self.M0 = device_put(jnp.array(-19.5))
        self.RV_MW = device_put(jnp.array(3.1))

        self.scale = 1e18
        self.device_scale = device_put(jnp.array(self.scale))
        self.sigma_pec = device_put(jnp.array(150 / 3e5))

        try:
            self.l_knots = np.genfromtxt(f'model_files/{load_model}/l_knots.txt')
            self.tau_knots = np.genfromtxt(f'model_files/{load_model}/tau_knots.txt')
            self.W0 = np.genfromtxt(f'model_files/{load_model}/W0.txt')
            self.W1 = np.genfromtxt(f'model_files/{load_model}/W1.txt')
            self.L_Sigma = np.genfromtxt(f'model_files/{load_model}/L_Sigma_epsilon.txt')
            model_params = np.genfromtxt(f'model_files/{load_model}/M0_sigma0_RV_tauA.txt')
            self.sigma0 = device_put(model_params[1])
            self.Rv = device_put(model_params[2])
            self.tauA = device_put(model_params[3])
        except:
            raise ValueError('Must select one of M20_model, T21_model, T21_partial-split_model and W22_model')

        self.l_knots = device_put(self.l_knots)
        self.tau_knots = device_put(self.tau_knots)
        self.W0 = device_put(self.W0)
        self.W1 = device_put(self.W1)
        self.L_Sigma = device_put(self.L_Sigma)

        # Initialise arrays and values for band responses - these are based on ParSNiP as presented in Boone+22
        self.min_wave = self.l_knots[0]
        self.max_wave = self.l_knots[-1]
        self.spectrum_bins = 300
        self.band_oversampling = 51
        self.max_redshift = 4

        self.obsmode_file = obsmodel_file
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
        self.KD_t_hsiao = device_put(spline_utils.invKD_irr(hsiao_phase))
        self.J_l_T_hsiao = device_put(spline_utils.spline_coeffs_irr(self.model_wave,
                                                                     hsiao_wave, KD_l_hsiao))

        self.hsiao_t = device_put(hsiao_phase)
        self.hsiao_l = device_put(hsiao_wave)
        self.hsiao_flux = device_put(hsiao_flux.T)
        self.hsiao_flux = jnp.matmul(self.J_l_T_hsiao, self.hsiao_flux)

    def _setup_band_weights(self):
        """Setup the interpolation for the band weights used for photometry"""
        # Build the model in log wavelength
        model_log_wave = np.linspace(np.log10(self.min_wave),
                                     np.log10(self.max_wave),
                                     self.spectrum_bins)

        model_spacing = model_log_wave[1] - model_log_wave[0]

        band_spacing = model_spacing / self.band_oversampling
        band_max_log_wave = (
                np.log10(self.max_wave * (1 + self.max_redshift))
                + band_spacing
        )

        # Oversampling must be odd.
        assert self.band_oversampling % 2 == 1
        pad = (self.band_oversampling - 1) // 2
        band_log_wave = np.arange(np.log10(self.min_wave),
                                  band_max_log_wave, band_spacing)
        band_wave = 10 ** (band_log_wave)
        band_pad_log_wave = np.arange(
            np.log10(self.min_wave) - band_spacing * pad,
            band_max_log_wave + band_spacing * pad,
            band_spacing
        )

        # Load reference source spectra
        with fits.open('data/zp/alpha_lyr_stis_010.fits') as hdu:
            vega_df = pd.DataFrame.from_records(hdu[1].data)
        vega_lam, vega_f = vega_df.WAVELENGTH, vega_df.FLUX

        def f_lam(l):
            f = (const.c.to('AA/s').value / 1e23) * ((l) ** -2) * 10 ** (-48.6 / 2.5) * 1e23
            return f

        band_weights, zps = [], []
        self.band_dict, self.zp_dict = {}, {}

        obsmode = pd.read_csv(self.obsmode_file, delim_whitespace=True)

        band_ind = 0
        for i, row in obsmode.iterrows():
            band, magsys = row.pb, row.magsys
            try:
                R = np.loadtxt(os.path.join('data', row.obsmode))
            except:
                continue
            band_transmission = np.interp(10 ** band_pad_log_wave, R[:, 0], R[:, 1])

            # Convolve the bands to match the sampling of the spectrum.
            band_conv_transmission = jnp.interp(band_wave, 10 ** band_pad_log_wave, band_transmission)

            dlamba = jnp.diff(band_wave)
            dlamba = jnp.r_[dlamba, dlamba[-1]]

            num = band_wave * band_conv_transmission * dlamba
            denom = jnp.sum(num)
            band_weight = num / denom

            band_weights.append(band_weight)

            # Get zero points
            lam = R[:, 0]
            if row.magsys == 'abmag':
                zp = f_lam(lam)
            elif row.magsys == 'vegamag':
                zp = interp1d(vega_lam, vega_f, kind='cubic')(lam)
            else:
                continue

            int1 = simpson(lam * zp * R[:, 1], lam)
            int2 = simpson(lam * R[:, 1], lam)
            zp = 2.5 * np.log10(int1 / int2)
            self.band_dict[band] = band_ind
            self.zp_dict[band] = zp
            zps.append(zp)
            band_ind += 1

        self.zps = zps

        # Get the locations that should be sampled at redshift 0. We can scale these to
        # get the locations at any redshift.
        band_interpolate_locations = jnp.arange(
            0,
            self.spectrum_bins * self.band_oversampling,
            self.band_oversampling
        )

        # Save the variables that we need to do interpolation.
        self.band_interpolate_locations = device_put(band_interpolate_locations)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = jnp.array(band_weights)
        self.model_wave = 10 ** (model_log_wave)

    def _calculate_band_weights(self, redshifts, ebv):
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

        # Hack fix maybe
        sum = jnp.sum(result, axis=1)
        result /= sum[:, None, :]

        # Apply MW extinction
        abv = self.RV_MW * ebv
        mw_array = jnp.zeros((result.shape[0], result.shape[1]))
        for i, val in enumerate(abv):
            mw = jnp.power(10, -0.4 * extinction.fitzpatrick99(self.model_wave * (1 + np.array(redshifts[i])), val, self.RV_MW))
            mw_array = mw_array.at[i, :].set(mw)

        result = result * mw_array[..., None]

        # We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        return result

    def get_spectra(self, theta, Av, W0, W1, eps, Rv, J_t, hsiao_interp):
        num_batch = theta.shape[0]
        W0 = jnp.repeat(W0[None, ...], num_batch, axis=0)
        W1 = jnp.repeat(W1[None, ...], num_batch, axis=0)

        W = W0 + theta[..., None, None] * W1 + eps

        WJt = jnp.matmul(W, J_t)
        W_grid = jnp.matmul(self.J_l_T, WJt)

        low_hsiao = self.hsiao_flux[:, hsiao_interp[0, ...].astype(int)]
        up_hsiao = self.hsiao_flux[:, hsiao_interp[1, ...].astype(int)]
        H_grid = ((1 - hsiao_interp[2, :]) * low_hsiao + hsiao_interp[2, :] * up_hsiao).transpose(2, 0, 1)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)

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

        A = Av[..., None] * (1 + (self.M_fitz_block @ yk.T).T / Rv[..., None])  # Rv[..., None]
        f_A = 10 ** (-0.4 * A)
        model_spectra = model_spectra * f_A[..., None]

        return model_spectra

    def get_flux_batch(self, theta, Av, W0, W1, eps, Ds, Rv, band_indices, flag, J_t, hsiao_interp, weights):
        num_batch = theta.shape[0]
        num_observations = band_indices.shape[0]

        model_spectra = self.get_spectra(theta, Av, W0, W1, eps, Rv, J_t, hsiao_interp)

        batch_indices = (
            jnp.arange(num_batch)
            .repeat(num_observations)
        ).astype(int)

        obs_band_weights = (
            weights[batch_indices, :, band_indices.T.flatten()]
            .reshape((num_batch, num_observations, -1))
            .transpose(0, 2, 1)
        )

        model_flux = jnp.sum(model_spectra * obs_band_weights, axis=1).T
        model_flux = model_flux * 10 ** (-0.4 * (self.M0 + Ds))
        model_flux *= self.device_scale
        model_flux *= flag
        return model_flux

        zps = self.zp[band_indices]
        model_mag = self.M0 + Ds - 2.5 * jnp.log10(model_flux / zps)
        model_mag *= flag

        return model_mag

    @staticmethod
    def spline_coeffs_irr_step(x_now, x, invkd):
        X = jnp.zeros_like(x)
        up_extrap = x_now > x[-1]
        down_extrap = x_now < x[0]
        interp = 1 - up_extrap - down_extrap

        h = x[-1] - x[-2]
        a = (x[-1] - x_now) / h
        b = 1 - a
        f = (x_now - x[-1]) * h / 6.0

        X = X.at[-2].set(X[-2] + a * up_extrap)
        X = X.at[-1].set(X[-1] + b * up_extrap)
        X = X.at[:].set(X[:] + f * invkd[-2, :] * up_extrap)

        h = x[1] - x[0]
        b = (x_now - x[0]) / h
        a = 1 - b
        f = (x_now - x[0]) * h / 6.0

        X = X.at[0].set(X[0] + a * down_extrap)
        X = X.at[1].set(X[1] + b * down_extrap)
        X = X.at[:].set(X[:] - f * invkd[1, :] * down_extrap)

        q = jnp.argmax(x_now < x) - 1
        h = x[q + 1] - x[q]
        a = (x[q + 1] - x_now) / h
        b = 1 - a
        c = ((a ** 3 - a) / 6) * h ** 2
        d = ((b ** 3 - b) / 6) * h ** 2

        X = X.at[q].set(X[q] + a * interp)
        X = X.at[q + 1].set(X[q + 1] + b * interp)
        X = X.at[:].set(X[:] + c * invkd[q, :] * interp + d * invkd[q + 1, :] * interp)

        return X

    def fit_model(self, obs, weights):
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            Av = numpyro.sample(f'AV', dist.Exponential(1 / self.tauA))
            # Rv = numpyro.sample('Rv', dist.Uniform(1, 6))
            tmax = numpyro.sample('tmax', dist.Uniform(-5, 5))
            t = obs[0, ...] - tmax[None, sn_index]
            hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1, 2, 0)
            eps_mu = jnp.zeros(N_knots_sig)
            # eps = numpyro.sample('eps', dist.MultivariateNormal(eps_mu, scale_tril=self.L_Sigma))
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            # eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 10
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err)) # Ds_err
            flux = self.get_flux_batch(theta, Av, self.W0, self.W1, eps, Ds, self.Rv, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)  # _{sn_index}

    def fit(self, num_samples, num_warmup, num_chains, output, result_path, chain_method='parallel', init_strategy='median'):
        if init_strategy == 'value':
            init_strategy = init_to_value(values=self.initial_guess())
        elif init_strategy == 'median':
            init_strategy = init_to_median()
        elif init_strategy == 'sample':
            init_strategy = init_to_sample()
        else:
            raise ValueError('Invalid init strategy, must be one of value, median and sample')
        if result_path != 'T21':
            with open(os.path.join('results', result_path, 'chains.pkl'), 'rb') as file:
                result = pickle.load(file)
            self.W0 = device_put(
                np.reshape(np.mean(result['W0'], axis=(0, 1)), (self.l_knots.shape[0], self.tau_knots.shape[0]),
                           order='F'))
            self.W1 = device_put(
                np.reshape(np.mean(result['W1'], axis=(0, 1)), (self.l_knots.shape[0], self.tau_knots.shape[0]),
                           order='F'))
            #sigmaepsilon = np.mean(result['sigmaepsilon'], axis=(0, 1))
            #L_Omega = np.mean(result['L_Omega'], axis=(0, 1))
            #self.L_Sigma = device_put(jnp.matmul(jnp.diag(sigmaepsilon), L_Omega))
            self.Rv = device_put(np.mean(result['Rv'], axis=(0, 1)))
            self.sigma0 = device_put(np.mean(result['sigma0'], axis=(0, 1)))
            self.tauA = device_put(np.mean(result['tauA'], axis=(0, 1)))

        #self.band_weights = self._calculate_band_weights(self.data[-5, 0, :], self.data[-2, 0, :])
        self.data = self.data[..., 41:43]
        self.band_weights = self.band_weights[41:43, ...]

        rng = PRNGKey(321)
        nuts_kernel = NUTS(self.fit_model, adapt_step_size=True, init_strategy=init_strategy, max_tree_depth=10)

        self.J_t_map = jax.jit(jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None)))

        def do_mcmc(data, weights):
            rng_key = PRNGKey(123)
            nuts_kernel = NUTS(self.fit_model, adapt_step_size=True, init_strategy=init_strategy,
                               max_tree_depth=10)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains,
                        chain_method=chain_method, progress_bar=True)
            mcmc.run(rng_key, data[..., None], weights[None, ...])
            return {**mcmc.get_samples(group_by_chain=True), **mcmc.get_extra_fields(group_by_chain=True)}

        map = jax.vmap(do_mcmc, in_axes=(2, 0))
        start = timeit.default_timer()
        samples = map(self.data, self.band_weights)
        for key, val in samples.items():
            val = np.squeeze(val)
            if len(val.shape) == 4:
                samples[key] = val.transpose(1, 2, 0, 3)
            else:
                samples[key] = val.transpose(1, 2, 0)
        end = timeit.default_timer()
        print('vmap: ', end - start)
        self.fit_postprocess(samples, output)

        start = timeit.default_timer()
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains,
                    chain_method=chain_method)
        mcmc.run(rng, self.data, self.band_weights)
        mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        end = timeit.default_timer()
        print('original: ', end - start)
        #self.fit_postprocess(samples, output)

    def fit_postprocess(self, samples, output):
        if not os.path.exists(os.path.join('results', output)):
            os.mkdir(os.path.join('results', output))

        muhat = self.data[-3, 0, :]
        muhat_err = 10
        Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
        mu = np.random.normal((samples['Ds'] * np.power(muhat_err, 2) + muhat * np.power(self.sigma0, 2)) /
                              np.power(Ds_err, 2),
                              np.sqrt((np.power(self.sigma0, 2) * np.power(muhat_err, 2)) / np.power(Ds_err, 2)))
        delM = samples['Ds'] - mu
        samples['mu'] = mu
        samples['delM'] = delM
        with open(os.path.join('results', output, 'chains.pkl'), 'wb') as file:
            pickle.dump(samples, file)
        # Save convergence data for each parameter to csv file
        summary = arviz.summary(samples)
        summary.to_csv(os.path.join('results', output, 'fit_summary.csv'))

    def train_model(self, obs):
        sample_size = self.data.shape[-1]
        N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        W_mu = jnp.zeros(N_knots)
        W0 = numpyro.sample('W0', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W1 = numpyro.sample('W1', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W0 = jnp.reshape(W0, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = jnp.reshape(W1, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')

        # sigmaepsilon = numpyro.sample('sigmaepsilon', dist.HalfNormal(1 * jnp.ones(N_knots_sig)))
        sigmaepsilon_tform = numpyro.sample('sigmaepsilon_tform', dist.Uniform(0, (jnp.pi / 2.) * jnp.ones(N_knots_sig)))
        sigmaepsilon = numpyro.deterministic('sigmaepsilon', 1. * jnp.tan(sigmaepsilon_tform))
        L_Omega = numpyro.sample('L_Omega', dist.LKJCholesky(N_knots_sig))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon), L_Omega)

        # sigma0 = numpyro.sample('sigma0', dist.HalfCauchy(0.1))
        sigma0_tform = numpyro.sample('sigma0_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0 = numpyro.deterministic('sigma0', 0.1 * jnp.tan(sigma0_tform))
        
        Rv = numpyro.sample('Rv', dist.Uniform(1, 5))
        # tauA = numpyro.sample('tauA', dist.HalfCauchy())
        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            Av = numpyro.sample(f'AV', dist.Exponential(1 / tauA))

            eps_mu = jnp.zeros(N_knots_sig)
            # eps = numpyro.sample('eps', dist.MultivariateNormal(eps_mu, scale_tril=L_Sigma))
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            #eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                       self.band_weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

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
        param_init = {}
        tauA_ = tauA_init + np.random.normal(0, 0.01)
        while tauA_ < 0:
            tauA_ = tauA_init + np.random.normal(0, 0.01)
        sigma0_ = sigma0_init + np.random.normal(0, 0.01)
        param_init['W0'] = jnp.array(W0_init + np.random.normal(0, 0.01, W0_init.shape[0]))
        param_init['W1'] = jnp.array(W1_init + np.random.normal(0, 0.01, W1_init.shape[0]))
        param_init['Rv'] = jnp.array(RV_init + np.random.uniform(1.0 - RV_init, 5.0 - RV_init))
        param_init['tauA_tform'] = jnp.arctan(tauA_ / 1.)
        # param_init['tauA'] = tauA_
        param_init['sigma0_tform'] = jnp.arctan(sigma0_ / 0.1)
        param_init['sigma0'] = jnp.array(sigma0_)
        param_init['theta'] = jnp.array(np.random.normal(0, 1, n_sne))
        param_init['Av'] = jnp.array(np.random.exponential(tauA_, n_sne))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon_init), L_Omega_init)

        with open(os.path.join('results', 'foundation_train_1000_val', 'chains.pkl'), 'rb') as file:
            chains = pickle.load(file)

        #param_init['theta'] = device_put(chains['theta'].mean(axis=(0, 1)))
        #param_init['Av'] = device_put(chains['AV'].mean(axis=(0, 1)))

        param_init['epsilon_tform'] = jnp.matmul(np.linalg.inv(L_Sigma), np.random.normal(0, 1, (n_eps, n_sne)))
        param_init['epsilon'] = np.random.normal(0, 1, (n_sne, n_eps))
        param_init['sigmaepsilon_tform'] = jnp.arctan(
            sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape) / 1.)
        param_init['sigmaepsilon'] = sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape)
        param_init['L_Omega'] = jnp.array(L_Omega_init)

        param_init['Ds'] = jnp.array(np.random.normal(self.data[-3, 0, :], sigma0_))

        return param_init

    def train(self, num_samples, num_warmup, num_chains, output, chain_method='parallel', init_strategy='median', mode='flux',
              l_knots=None):
        if init_strategy == 'value':
            init_strategy = init_to_value(values=self.initial_guess())
        elif init_strategy == 'median':
            init_strategy = init_to_median()
        elif init_strategy == 'sample':
            init_strategy = init_to_sample()
        else:
            raise ValueError('Invalid init strategy, must be one of value, median and sample')
        if l_knots is not None:
            self.l_knots = jnp.array(l_knots)
            KD_l = spline_utils.invKD_irr(self.l_knots)
            self.J_l_T = device_put(spline_utils.spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        self.band_weights = self._calculate_band_weights(self.data[-5, 0, :], self.data[-2, 0, :])
        self.zp = jnp.array([4.608419288004386e-09, 2.8305383925373084e-09, 1.917161265703195e-09, 1.446643295845274e-09])
        t = self.data[0, ...]
        self.hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        rng = PRNGKey(321)
        # rng = jnp.array([PRNGKey(11), PRNGKey(22), PRNGKey(33), PRNGKey(44)])
        #rng = PRNGKey(101)
        # numpyro.render_model(self.train_model, model_args=(self.data,), filename='train_model.pdf')
        nuts_kernel = NUTS(self.train_model, adapt_step_size=True, target_accept_prob=0.8, init_strategy=init_strategy,
                           dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains,
                    chain_method=chain_method)
        mcmc.run(rng, self.data, extra_fields=('potential_energy',))
        mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        extras = mcmc.get_extra_fields(group_by_chain=True)
        self.train_postprocess(samples, extras, output)

    def train_postprocess(self, samples, extras, output):
        if not os.path.exists(os.path.join('results', output)):
            os.mkdir(os.path.join('results', output))
        with open(os.path.join('results', output, 'initial_chains.pkl'), 'wb') as file:
            pickle.dump(samples, file)
        # Sign flipping-----------------
        J_R = spline_utils.spline_coeffs_irr([6200.0], self.l_knots, spline_utils.invKD_irr(self.l_knots))
        J_10 = spline_utils.spline_coeffs_irr([10.0], self.tau_knots, spline_utils.invKD_irr(self.tau_knots))
        J_0 = spline_utils.spline_coeffs_irr([0.0], self.tau_knots, spline_utils.invKD_irr(self.tau_knots))
        W1 = np.reshape(samples['W1'], (samples['W1'].shape[0], samples['W1'].shape[1], self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        N_chains = W1.shape[0]
        sign = np.zeros(N_chains)
        for chain in range(N_chains):
            chain_W1 = np.mean(W1[chain, ...], axis=0)
            chain_sign = np.sign(
                np.squeeze(np.matmul(J_R, np.matmul(chain_W1, J_10.T))) - np.squeeze(np.matmul(J_R, np.matmul(chain_W1, J_0.T))))
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

        #sigmaepsilon = np.mean(samples['sigmaepsilon'], axis=[0, 1])
        #L_Omega = np.mean(samples['L_Omega'], axis=[0, 1])
        #L_Sigma = np.matmul(np.diag(np.mean(samples['sigmaepsilon'], axis=[0, 1])), np.mean(samples['L_Omega'], axis=[0, 1]))
        sigma0 = np.mean(samples['sigma0'])

        Rv = np.mean(samples['Rv'])
        tauA = np.mean(samples['tauA'])
        M0_sigma0_RV_tauA = np.array([self.M0, sigma0, Rv, tauA])
        np.savetxt(os.path.join('results', output, 'W0.txt'), W0, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'W1.txt'), W1, delimiter="\t", fmt="%.3f")
        #np.savetxt(os.path.join('results', output, 'sigmaepsilon.txt'), sigmaepsilon, delimiter="\t", fmt="%.3f")
        #np.savetxt(os.path.join('results', output, 'L_Omega.txt'), L_Omega, delimiter="\t", fmt="%.3f")
        #np.savetxt(os.path.join('results', output, 'L_Sigma.txt'), L_Sigma, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'M0_sigma0_RV_tauA.txt'), M0_sigma0_RV_tauA, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'l_knots.txt'), self.l_knots, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'tau_knots.txt'), self.tau_knots, delimiter="\t", fmt="%.3f")

        # Save extra fields
        potentials = extras['potential_energy']
        np.savetxt(os.path.join('results', output, 'potentials.txt'), potentials)

        """global_param_dict = {
            'W0': repr(np.round(np.mean(samples['W0'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]), order='F'), 3).tolist()),
            'W1': repr(np.mean(samples['W1'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]),
                                                              order='F').tolist()),
            'sigmaepsilon': repr(np.mean(samples['sigmaepsilon'], axis=[0, 1])),
            'L_Omega': repr(np.mean(samples['L_Omega'], axis=[0, 1])),
            'L_Sigma': repr(np.matmul(np.diag(np.mean(samples['sigmaepsilon'], axis=[0, 1])), np.mean(samples['L_Omega'], axis=[0, 1]))),
            'sigma0': np.mean(samples['sigma0'])
        }"""

    def process_dataset(self, sample_name, lc_dir, meta_file, map_dict=None, sn_list=None, data_mode='flux'):
        if not os.path.exists(os.path.join('data', 'LCs', 'pickles', sample_name)):
            os.mkdir(os.path.join('data', 'LCs', 'pickles', sample_name))
        if os.path.exists(os.path.join('data', 'LCs', 'pickles', sample_name, f'dataset_{data_mode}.pkl')):
            with open(os.path.join('data', 'LCs', 'pickles', sample_name, f'dataset_{data_mode}.pkl'), 'rb') as file:
                all_data = pickle.load(file)
            with open(os.path.join('data', 'LCs', 'pickles', sample_name, 'J_t.pkl'), 'rb') as file:
                all_J_t = pickle.load(file)
            self.data = device_put(all_data.T)
            self.J_t = device_put(all_J_t)
            self.band_weights = self._calculate_band_weights(self.data[-5, 0, :], self.data[-2, 0, :])
            return
        if sn_list is not None and sample_name.lower() == 'foundation':
            sn_list = pd.read_csv(sn_list, names=['file'])
            sn_list['sn'] = sn_list.file.apply(lambda x: x[x.rfind('_') + 1: x.rfind('.')])
        elif sn_list is not None:
            sn_list = pd.read_csv(sn_list, names=['file'])
            sn_list['sn'] = sn_list.file.apply(lambda x: x[:x.find('.')])
        else:
            sn_list = pd.DataFrame(os.listdir(lc_dir), columns=['file'])
            sn_list['sn'] = sn_list.file.apply(lambda x: x[:x.find('.')])

        meta_file = pd.read_csv(meta_file, delim_whitespace=True)
        sn_list = sn_list.merge(meta_file, left_on='sn', right_on='SNID')
        n_obs = []

        all_lcs = []
        t_ranges = []
        for i, row in sn_list.iterrows():
            meta, lcdata = sncosmo.read_snana_ascii(os.path.join(lc_dir, row.file), default_tablename='OBS')
            data = lcdata['OBS'].to_pandas()
            data['t'] = (data.MJD - row.SEARCH_PEAKMJD) / (1 + row.REDSHIFT_CMB)
            if map_dict is not None:
                data['band_indices'] = data.FLT.apply(lambda x: self.band_dict[map_dict[x]])
                data['zp'] = data.FLT.apply(lambda x: self.zp_dict[map_dict[x]])
            else:
                data['band_indices'] = data.FLT.apply(lambda x: self.band_dict[x])
                data['zp'] = data.FLT.apply(lambda x: self.zp_dict[x])
            data['flux'] = np.power(10, -0.4 * (data['MAG'] - data['zp'])) * self.scale
            data['flux_err'] = (np.log(10) / 2.5) * data['flux'] * data['MAGERR']
            data['redshift'] = row.REDSHIFT_CMB
            data['redshift_error'] = row.REDSHIFT_CMB_ERR
            data['MWEBV'] = meta['MWEBV']
            data['dist_mod'] = self.cosmo.distmod(row.REDSHIFT_CMB)
            data['flag'] = 1
            if data_mode == 'flux':
                lc = data[['t', 'flux', 'flux_err', 'band_indices', 'redshift', 'redshift_error', 'dist_mod', 'MWEBV', 'flag']]
                lc = lc.dropna(subset=['flux', 'flux_err'])
            else:
                lc = data[['t', 'MAG', 'MAGERR', 'band_indices', 'redshift', 'redshift_error', 'dist_mod', 'MWEBV',
                           'flag']]
                lc = lc.dropna(subset=['MAG', 'MAGERR'])
            lc = lc[(lc['t'] > -10) & (lc['t'] < 40)]
            t_ranges.append((lc['t'].min(), lc['t'].max()))
            n_obs.append(lc.shape[0])
            all_lcs.append(lc)
        N_sn = sn_list.shape[0]
        N_obs = np.max(n_obs)
        N_col = lc.shape[1]
        all_data = np.zeros((N_sn, N_obs, N_col))
        all_J_t = np.zeros((N_sn, self.tau_knots.shape[0], N_obs))
        all_J_t_hsiao = np.zeros((N_sn, self.hsiao_t.shape[0], N_obs))
        for i, lc in enumerate(all_lcs):
            all_data[i, :lc.shape[0], :] = lc.values
            all_data[i, lc.shape[0]:, 2] = 1 / jnp.sqrt(2 * np.pi)
            all_J_t[i, ...] = spline_utils.spline_coeffs_irr(all_data[i, :, 0], self.tau_knots, self.KD_t).T
        with open(os.path.join('data', 'LCs', 'pickles', sample_name, f'dataset_{data_mode}.pkl'), 'wb') as file:
            pickle.dump(all_data, file)
        with open(os.path.join('data', 'LCs', 'pickles', sample_name, 'J_t.pkl'), 'wb') as file:
            pickle.dump(all_J_t, file)
        self.data = device_put(all_data.T)
        self.J_t = device_put(all_J_t)
        self.J_t_hsiao = device_put(all_J_t_hsiao)
        self.band_weights = self._calculate_band_weights(self.data[-5, 0, :], self.data[-2, 0, :])

    def fit_from_results(self, input_file):
        with open(os.path.join('results', f'{input_file}.pkl'), 'rb') as file:
            result = pickle.load(file)
        N = result['theta'].shape[1]
        W0 = np.reshape(np.mean(result['W0'], axis=0), (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = np.reshape(np.mean(result['W1'], axis=0), (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        eps = np.reshape(np.mean(result['eps'], axis=0), (N, 4, 6), order='F')
        new_eps = np.zeros((eps.shape[0], eps.shape[1] + 2, eps.shape[2]))
        theta, Av, Ds = np.mean(result['theta'], axis=0), np.mean(result['AV'], axis=0), np.mean(result['Ds'], axis=0)
        new_eps[:, 1:-1, :] = eps
        eps = new_eps
        N_fit = self.J_t.shape[-1]
        band_indices = self.data[-4, :, :].astype(int)
        fit_band_indices = np.tile(np.arange(4), (band_indices.shape[-1], int(N_fit / len(self.band_dict.keys())))).T
        redshift = self.data[-3, 0, :]
        flag = self.data[-1, ...]
        fit_flag = np.ones_like(fit_band_indices)
        model_flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, redshift, fit_band_indices, fit_flag)
        ts = np.linspace(-10, 40, 50)
        for _ in range(10):
            plt.figure()
            for i in range(4):
                inds = (band_indices[:, _] == i) & (flag[:, _] == 1)
                fit_inds = (fit_band_indices[:, _] == i) & (fit_flag[:, _] == 1)
                plt.errorbar(self.data[0, inds, _], self.data[1, inds, _], yerr=self.data[2, inds, _], fmt='x')
                plt.plot(ts, model_flux[fit_inds, _], ls='--')
        plt.show()

    def get_flux_from_chains(self, model):
        with open(os.path.join('results', model, 'chains.pkl'), 'rb') as file:
            chains = pickle.load(file)
        sample_size = chains['theta'].shape[-1]
        self.band_weights = self._calculate_band_weights(jnp.zeros(sample_size), jnp.zeros(sample_size))
        for param, samples in chains.items():
            chains[param] = samples.reshape((samples.shape[0] * samples.shape[1], *samples.shape[2:]), order='F')

        t = np.arange(-8, 40, 4)[..., None]
        steps_per_band = t.shape[0]
        t = jnp.tile(t, (4, sample_size))
        keep_shape = t.shape
        t = t.flatten(order='F')
        map = jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None))
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1, 2, 0)
        J_t_hsiao = map(t, self.hsiao_t, self.KD_t_hsiao).reshape((*keep_shape, self.hsiao_t.shape[0]),
                                                                  order='F').transpose(1, 2, 0)
        t = t.reshape(keep_shape, order='F')
        band_indices = jnp.tile(np.array([[i] * 12 for i in range(4)]).flatten()[..., None], (1, sample_size))
        flag = jnp.ones_like(band_indices)

        eps = chains['eps']
        eps = np.reshape(chains['eps'], (eps.shape[0], self.l_knots.shape[0] - 2, self.tau_knots.shape[0], sample_size),
                         order='F')
        eps_full = jnp.zeros((eps.shape[0], self.l_knots.shape[0], self.tau_knots.shape[0], sample_size))
        eps = eps_full.at[:, 1:-1, :, :].set(eps).transpose(0, 3, 1, 2)
        theta = device_put(chains['theta'])
        Av = device_put(chains['AV'] * 0)
        Ds = device_put(chains['Ds'])
        Rv = jnp.array(2.61)

        jit_flux_batch = jax.jit(self.get_flux_batch)
        jit_flux_batch(theta[0, ...], Av[0, ...], self.W0, self.W1, eps[0, ...], Ds[0, ...], Rv, band_indices,
                            flag, J_t, J_t_hsiao)
        map = jax.vmap(jit_flux_batch, in_axes=(0, 0, None, None, 0, 0, None, None, None, None, None))
        flux = map(theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, flag, J_t, J_t_hsiao) / self.device_scale
        flux_bands = np.zeros((flux.shape[0], int(flux.shape[1] / steps_per_band), steps_per_band, flux.shape[-1]))
        for i in range(int(flux.shape[1] / steps_per_band)):
            flux_bands[:, i, ...] = flux[:, i * steps_per_band: (i + 1) * steps_per_band, ...]
        self.zp = np.array(
            [4.608419288004386e-09, 2.8305383925373084e-09, 1.917161265703195e-09, 1.446643295845274e-09])
        mag_bands = -2.5 * np.log10(flux_bands / self.zp[None, :, None, None])
        np.save(os.path.join('results', model, 'rf_mags'), mag_bands)

        eps = eps * 0

        flux = map(theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, flag, J_t, J_t_hsiao) / self.device_scale
        flux_bands = np.zeros((flux.shape[0], int(flux.shape[1] / steps_per_band), steps_per_band, flux.shape[-1]))
        for i in range(int(flux.shape[1] / steps_per_band)):
            flux_bands[:, i, ...] = flux[:, i * steps_per_band: (i + 1) * steps_per_band, ...]
        self.zp = np.array(
            [4.608419288004386e-09, 2.8305383925373084e-09, 1.917161265703195e-09, 1.446643295845274e-09])
        mag_bands = -2.5 * np.log10(flux_bands / self.zp[None, :, None, None])
        np.save(os.path.join('results', model, 'rf_mags_eps0'), mag_bands)

    def plot_hubble_diagram(self, model):
        with open(os.path.join('results', model, 'chains.pkl'), 'rb') as file:
            chains = pickle.load(file)
        redshifts = np.array(self.data[-5, 0, :])
        mu_model = self.cosmo.distmod(redshifts).value
        mu_obs, mu_obs_err = chains['mu'].mean(axis=(0, 1)), chains['mu'].std(axis=(0, 1))
        theta, theta_err = chains['theta'].mean(axis=(0, 1)), chains['theta'].std(axis=(0, 1))
        Av, Av_err = chains['AV'].mean(axis=(0, 1)), chains['AV'].std(axis=(0, 1))
        plt.errorbar(redshifts, mu_obs, yerr=mu_obs_err, fmt='x')
        plt.plot(np.sort(redshifts), np.sort(mu_model), ls='--')
        plt.show()
        hres = mu_obs - mu_model
        hres_err = mu_obs_err
        plt.errorbar(redshifts, hres, yerr=hres_err, fmt='x')
        plt.show()
        save_data = np.array([redshifts, hres, hres_err, theta, theta_err, Av, Av_err])
        np.save(os.path.join('results', model, 'hres'), save_data)

    def compare_params(self):
        sn_list = pd.read_csv('data/LCs/foundation/Foundation_DR1/Foundation_DR1.LIST', names=['file'])
        sn_list['sn'] = sn_list.file.apply(lambda x: x[x.rfind('_') + 1: x.rfind('.')])
        meta_file = pd.read_csv('data/LCs/meta/T21_training_set_meta.txt', delim_whitespace=True)
        sn_list = sn_list.merge(meta_file, left_on='sn', right_on='SNID')
        T21_order = pd.read_csv(
            'model_files/T21_model/sn_training_list_realn157F_alpha_4x500+500_nou_av-exp_W1_210204_180221.txt',
            header=None, names=['sn'])
        T21_order['inds'] = np.arange(157)
        sn_list = sn_list.merge(T21_order, on='sn')
        ord = sn_list['inds'].values

        T21_summary = pd.read_csv('model_files/T21_model/summary_realn157F_alpha_4x500+500_nou_av-exp_W1_210204_180221.txt', delim_whitespace=True)
        # Reorder T21 params and prepare comparisons
        T21_summary = T21_summary.dropna()
        params = []
        for param in T21_summary.param.values:
            if 'epsilons' in param:
                num1, num2 = param[param.find('[') + 1:param.find(',')], param[param.find(',') + 1:-1]
                param = f'epsilons[{num2},{num1}]'
            params.append(param)
        T21_summary['param'] = params

        all_eps_df = None
        for i in ord + 1:
            eps_df = T21_summary[T21_summary.param.str.contains(f'epsilons\[{i},')]
            if eps_df is None:
                all_eps_df = eps_df
            else:
                all_eps_df = pd.concat([all_eps_df, eps_df])

        sn_param_dfs = []
        for param in ['AV', 'Ds']:
            param_df = T21_summary[T21_summary.param.str.contains(param)]
            param_df = param_df.iloc[ord, :]
            sn_param_dfs.append(param_df)

        glob_params_dfs = []
        for param in ['W1', 'sigmaepsilon', 'sigma0', 'RV']:
            param_df = T21_summary[T21_summary.param.str.contains(param)]
            glob_params_dfs.append(param_df)

        T21_comparison = pd.concat([*sn_param_dfs, *glob_params_dfs]).reset_index(drop=True)

        # Prepare numpyro comparisons
        np_summary = pd.read_csv('results/foundation_train_1000_mag/fit_summary.csv')
        np_summary.columns = ['param'] + list(np_summary.columns[1:])
        np_eps_df = np_summary[np_summary.param.str.contains('eps\[')]

        sn_param_dfs = []
        for param in ['AV', 'Ds']:
            param_df = np_summary[np_summary.param.str.contains(param)]
            sn_param_dfs.append(param_df)

        glob_params_dfs = []
        for param in ['W1', 'sigmaepsilon', 'sigma0', 'Rv']:
            param_df = np_summary[(np_summary.param.str.contains(param)) & (~np_summary.param.str.contains('tform'))]
            glob_params_dfs.append(param_df)

        np_comparison = pd.concat([*sn_param_dfs, *glob_params_dfs]).reset_index(drop=True)

        T21_comparison.columns = 'stan_' + T21_comparison.columns
        np_comparison.columns = 'np_' + np_comparison.columns
        comparison_df = T21_comparison.merge(np_comparison, left_index=True, right_index=True)

        """theta_df = comparison_df[comparison_df.stan_param.str.contains('Ds', case=True)]
        print(theta_df.np_mean.std())
        theta_err = np.sqrt(np.power(theta_df.stan_sd / np.sqrt(157), 2) + np.power(theta_df.np_sd / np.sqrt(157), 2))
        theta_sig = np.abs(theta_df.stan_mean - theta_df.np_mean) / theta_err
        theta_df['err'] = theta_err
        theta_df['sig'] = theta_sig
        print(theta_df.sig.describe(percentiles=(0.5, 0.68, 0.95)))
        # print(theta_df[['stan_mean', 'stan_sd', 'np_mean', 'np_sd']])
        plt.figure(figsize=(12, 8))
        plt.errorbar(theta_df['stan_mean'], theta_df['np_mean'], xerr=theta_df.stan_sd / np.sqrt(theta_df.stan_n_eff),
                     yerr=theta_df.np_sd / np.sqrt(theta_df.np_ess_bulk), fmt='x')
        #plt.plot([-5, 5], [-5, 5], ls='--')
        #plt.xlim(-0.5, 2)
        #plt.ylim(-0.5, 2)
        plt.xlabel(r'$W0$ (stan)')
        plt.ylabel(r'$W0$ (numpyro)')
        plt.show()
        plt.figure(figsize=(12, 8))
        plt.errorbar(theta_df['stan_mean'], theta_df['np_mean'] - theta_df['stan_mean'],
                     yerr=theta_df['err'], fmt='x')
        plt.xlabel('stan Ds')
        plt.ylabel('Residual')
        plt.show()
        return"""

        comparison_df['sig'] = np.abs(comparison_df.stan_mean - comparison_df.np_mean) / np.sqrt(np.power(comparison_df.stan_sd, 2) + np.power(comparison_df.np_sd, 2))
                               #np.sqrt(np.power(comparison_df.stan_sd / np.sqrt(comparison_df.stan_n_eff), 2) + np.power(comparison_df.np_sd / np.sqrt(comparison_df.np_ess_bulk), 2))
        # comparison_df = comparison_df[comparison_df.stan_param.str.contains('AV')]
        print(comparison_df.sig.describe())
        print(comparison_df.sort_values(by='sig', ascending=False))
        plt.figure(figsize=(12, 8))
        plt.hist(comparison_df.sig, bins=30)
        plt.xlabel('Significance of difference')
        plt.ylabel('Frequency')
        plt.show()

        return
        print(comparison_df.sig.describe())
        print(comparison_df.sort_values(by='sig', ascending=False).head(20))
        Rv1, Rv2 = T21_summary[T21_summary.param == 'RV'], np_summary[np_summary.param == 'Rv']
        Rv1_mcerr = Rv1.sd.values[0] / np.sqrt(Rv1.n_eff.values[0])
        Rv2_mcerr = Rv2.sd.values[0] / np.sqrt(Rv2.ess_bulk.values[0])
        Rv_diff = np.abs(Rv1['mean'].values[0] - Rv2['mean'].values[0])
        Rv_mcerr = np.sqrt(np.power(Rv1_mcerr, 2) + np.power(Rv2_mcerr, 2))
        Rv_sig = Rv_diff / Rv_mcerr
        print(Rv_sig)
        Av1, Av2 = T21_summary[T21_summary.param.str.contains('theta', case=False)], np_summary[np_summary.param.str.contains('theta', case=False)]
        plt.scatter(Av1['mean'].values[ord], Av2['mean'])
        plt.show()


# -------------------------------------------------

"""if __name__ == '__main__':
    model = Model()
    filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
    model.process_dataset('foundation', 'data/LCs/foundation/Foundation_DR1', 'data/LCs/meta/T21_training_set_meta.txt',
                          filt_map_dict, sn_list='data/LCs/foundation/Foundation_DR1/Foundation_DR1.LIST')
    #model.process_dataset('ztf', 'data/LCs/ZTF', 'data/LCs/meta/ztf_dr1_training.txt',
    #                      map_dict=None)
    #model.train(1000, 1000, 4, 'ztf_train_test', chain_method='parallel', init_strategy='median',
    #            l_knots=[4150, 4760, 6390, 7930, 9000])
    model.fit(250, 250, 4, 'foundation_fit_7tree', 'T21', chain_method='parallel')
    # model.get_flux_from_chains('foundation_fit_T21freeRv')
    # model.plot_hubble_diagram('foundation_fit_T21freeRv')
    # model.compare_params()
    # model.simulate_spectrum()
    # model.train_postprocess()
    # result.print_summary()
    # model.save_results_to_yaml(result, 'foundation_train_4chain')
    # model.fit_assess(params, '4chain_fit_test')
    # model.fit_from_results('foundation_train')
    # model.train_assess(params, 'gpu_train_dist')"""
