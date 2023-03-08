"""
BayeSN SED Model. Defines a class which allows you to fit or simulate from the
BayeSN Optical+NIR SED model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_sample, init_to_value
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import h5py
import sncosmo
import spline_utils
import pickle
import pandas as pd
import jax
from jax import device_put
import jax.numpy as jnp
from jax.random import PRNGKey
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import matplotlib as mpl
from matplotlib import rc
import arviz
import extinction
import timeit
from astropy.io import fits

# Make plots look pretty
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})

# jax.config.update('jax_platform_name', 'cpu')


class SEDmodel(object):
    """
    BayeSN-SED Model

    Class which imports a BayeSN model, and allows one to fit or simulate
    Type Ia supernovae based on this model.

    Parameters
    ----------
    model : str, optional
        Can be either a pre-defined BayeSN model name (see table below), or
        a path to directory containing a set of .txt files from which a
        valid model can be constructed. Currently implemented default models
        are listed below - default is M20. See README in `BayeSNmodel/model_files`
        for more info.
          ``'M20'`` | Mandel+20 BayeSN model (arXiv:2008.07538). Covers
                    |   rest wavelength range of 3000-18500A (BVRIYJH). No
                    |   treatment of host mass effects. Global RV assumed.
                    |   Trained on low-z Avelino+19 (ApJ, 887, 106)
                    |   compilation of CfA, CSP and others.
          ``'T21'`` | Thorp+21 No-Split BayeSN model (arXiv:2102:05678). Covers
                    |   rest wavelength range of 3500-9500A (griz). No
                    |   treatment of host mass effects. Global RV assumed.
                    |   Trained on Foundation DR1 (Foley+18, Jones+19).
    fiducial_cosmology :  dict, optional
        Dictionary containg kwargs ``{H0, Om0}`` for initialising a
        :py:class:`astropy.cosmology.FlatLambdaCDM` instance. Defaults to
        Riess+16 (ApJ, 826, 56) cosmology ``{H0:73.24, "Om0":0.28}``.

    Attributes
    ----------
    params : dict
        Dictionary containing BayeSN model parameters.
    zpt : float
        SNANA zero point which is to be assumed
    t_k : :py:class:`numpy.array`
        Array of time knots which the model is defined at
    l_k : :py:class:`numpy.array`
        Array of wavelength knots which the model is defined at
    knots : :py:class:`numpy.array`
        Cartesian product of t_k and l_k
    dl_int : float
        Wavelength pacing of the Hsiao template (defaults to 10A)
    hsiao : dict
        Dictionary containing specification of the Hsiao template from
        Hsiao+07 (ApJ, 663, 1187)
    passbands : dict
        Dictionary of all passbands with available specification. Indexed
        by filter name.
    cosmo : :py:class:`astropy.cosmology.FlatLambdaCDM`
        :py:class:`astropy.cosmology.FlatLambdaCDM` instance defining the
        fiducial cosmology which the model was trained using.
    stan_model : None or :py:class:`cmdstanpy.CmdStanModel`
        Compiled Stan model for fitting the BayeSN photometric distance
        model to a supernova.
    fixed_tmax : bool
        Indicates whether the currently compiled Stan model assumes a fixed
        time of maximum.
    fixed_RV : bool
        Indicates whether the currently compiled Stan model assumes a fixed RV
    scale: float
        Scaling factor used when training/fitting in flux space to ensure that flux values are of order unity

    Returns
    -------
    out : :py:class:`bayesn_model.SEDmodel` instance
    	"""

    def __init__(self, num_devices=8, enable_x64=True, load_model='T21_model',
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}, obsmodel_file='data/SNmodel_pb_obsmode_map.txt'):
        """

        Parameters
        ----------
        num_devices
        enable_x64
        load_model
        fiducial_cosmology
        obsmodel_file
        """
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
        """
        Loads the Hsiao template from the internal HDF5 file.

        Stores the template as an attribute of `SEDmodel`.

        """
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
        """
        Sets up the interpolation for the band weights used for photometry as well as calculating the zero points for
        each band. This code is partly based off ParSNiP from
        Boone+21
        """
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

        self.zps = jnp.array(zps)

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
        """
        Calculates the observer-frame band weights, including the effect of Milky Way extinction, for each SN

        Parameters
        ----------
        redshifts: array-like
            Array of redshifts for each SN
        ebv: array-like
            Array of Milky Way E(B-V) values for each SN

        Returns
        -------

        weights: array-like
            Array containing observer-frame band weights

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
        weights = flat_result.reshape((-1,) + locs.shape).transpose(1, 2, 0)

        # Normalise so max transmission = 1
        sum = jnp.sum(weights, axis=1)
        weights /= sum[:, None, :]

        # Apply MW extinction
        abv = self.RV_MW * ebv
        mw_array = jnp.zeros((weights.shape[0], weights.shape[1]))
        for i, val in enumerate(abv):
            mw = jnp.power(10, -0.4 * extinction.fitzpatrick99(self.model_wave * (1 + np.array(redshifts[i])), val,
                                                               self.RV_MW))
            mw_array = mw_array.at[i, :].set(mw)

        weights = weights * mw_array[..., None]

        # We need an extra term of 1 + z from the filter contraction.
        weights /= (1 + redshifts)[:, None, None]

        return weights

    def get_spectra(self, theta, Av, W0, W1, eps, Rv, J_t, hsiao_interp):
        """
        Calculates rest-frame spectra for given parameter values

        Parameters
        ----------
        theta: array-like
            Set of theta values for each SN
        Av: array-like
            Set of host extinction values for each SN
        W0: array-like
            Global W0 matrix
        W1: array-like
            Global W1 matrix
        eps: array-like
            Set of epsilon values for each SN, describing residual colour variation
        Rv: float
            Global R_V value for host extinction (need to allow this to be variable in future)
        J_t: array-like
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: array-like
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation


        Returns
        -------

        model_spectra: array-like
            Matrix containing model spectra for all SNe at all time-steps

        """
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

    def get_flux_batch(self, theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, J_t, hsiao_interp, weights):
        """
        Calculates observer-frame fluxes for given parameter values

        Parameters
        ----------
        theta: array-like
            Set of theta values for each SN
        Av: array-like
            Set of host extinction values for each SN
        W0: array-like
            Global W0 matrix
        W1: array-like
            Global W1 matrix
        eps: array-like
            Set of epsilon values for each SN, describing residual colour variation
        Ds: array-like
            Set of distance moduli for each SN
        Rv: float
            Global R_V value for host extinction (need to allow this to be variable in future)
        band_indices: array-like
            Array containing indices describing which filter each observation is in
        mask: array-like
            Array containing mask describing whether observations should contribute to the posterior
        J_t: array-like
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: array-like
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation
        weights: array_like
            Array containing band weights to use for photometry

        Returns
        -------

        model_flux: array-like
            Matrix containing model fluxes for all SNe at all time-steps

        """
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
        model_flux *= mask
        return model_flux

    def get_mag_batch(self, theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, J_t, hsiao_interp, weights):
        """
        Calculates observer-frame magnitudes for given parameter values

        Parameters
        ----------
        theta: array-like
            Set of theta values for each SN
        Av: array-like
            Set of host extinction values for each SN
        W0: array-like
            Global W0 matrix
        W1: array-like
            Global W1 matrix
        eps: array-like
            Set of epsilon values for each SN, describing residual colour variation
        Ds: array-like
            Set of distance moduli for each SN
        Rv: float
            Global R_V value for host extinction (need to allow this to be variable in future)
        band_indices: array-like
            Array containing indices describing which filter each observation is in
        mask: array-like
            Array containing mask describing whether observations should contribute to the posterior
        J_t: array-like
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: array-like
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation
        weights: array_like
            Array containing band weights to use for photometry

        Returns
        -------

        model_mag: array-like
            Matrix containing model magnitudes for all SNe at all time-steps
        """
        model_flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, J_t, hsiao_interp, weights)
        model_flux = model_flux / self.device_scale
        model_flux = model_flux + (1 - mask) * 0.01
        zps = self.zps[band_indices]

        model_mag = - 2.5 * jnp.log10(model_flux) + zps  # self.M0 + Ds
        model_mag *= mask

        return model_mag

    @staticmethod
    def spline_coeffs_irr_step(x_now, x, invkd):
        """
        Vectorized version of cubic spline coefficient calculator found in spline_utils

        Parameters
        ----------
        x_now: array-like
            Current x location to calculate spline knots for
        x: array-like
            Numpy array containing the locations of the spline knots.
        invkd: array-like
            Precomputed matrix for generating second derivatives. Can be obtained
            from the output of ``spline_utils.invKD_irr``.

        Returns
        -------

        X: Set of spline coefficients for each x knot

        """
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
        """

        Parameters
        ----------
        obs
        weights

        Returns
        -------

        """
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            Av = numpyro.sample(f'AV', dist.Exponential(1 / self.tauA))
            Rv = numpyro.sample('Rv', dist.Uniform(1, 6))
            tmax = numpyro.sample('tmax', dist.Uniform(-5, 5))
            t = obs[0, ...] - tmax[None, sn_index]
            hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]),
                                                                     order='F').transpose(1, 2, 0)
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
            muhat = obs[-3, 0, sn_index]
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 10
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))  # Ds_err
            flux = self.get_flux_batch(theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)  # _{sn_index}

    def fit(self, num_samples, num_warmup, num_chains, output, result_path, chain_method='parallel',
            init_strategy='median'):
        """

        Parameters
        ----------
        num_samples
        num_warmup
        num_chains
        output
        result_path
        chain_method
        init_strategy

        Returns
        -------

        """
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
            # sigmaepsilon = np.mean(result['sigmaepsilon'], axis=(0, 1))
            # L_Omega = np.mean(result['L_Omega'], axis=(0, 1))
            # self.L_Sigma = device_put(jnp.matmul(jnp.diag(sigmaepsilon), L_Omega))
            self.Rv = device_put(np.mean(result['Rv'], axis=(0, 1)))
            self.sigma0 = device_put(np.mean(result['sigma0'], axis=(0, 1)))
            self.tauA = device_put(np.mean(result['tauA'], axis=(0, 1)))

        # self.data = self.data[..., 41:42]
        # self.band_weights = self.band_weights[41:42, ...]

        rng = PRNGKey(321)
        nuts_kernel = NUTS(self.fit_model, adapt_step_size=True, init_strategy=init_strategy, max_tree_depth=10)

        self.J_t_map = jax.jit(jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None)))

        def do_mcmc(data, weights):
            """

            Parameters
            ----------
            data
            weights

            Returns
            -------

            """
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
        # self.fit_postprocess(samples, output)

    def fit_postprocess(self, samples, output):
        """

        Parameters
        ----------
        samples
        output

        Returns
        -------

        """
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
        """

        Parameters
        ----------
        obs

        Returns
        -------

        """
        sample_size = self.data.shape[-1]
        N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        W_mu = jnp.zeros(N_knots)
        W0 = numpyro.sample('W0', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W1 = numpyro.sample('W1', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W0 = jnp.reshape(W0, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = jnp.reshape(W1, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')

        # sigmaepsilon = numpyro.sample('sigmaepsilon', dist.HalfNormal(1 * jnp.ones(N_knots_sig)))
        sigmaepsilon_tform = numpyro.sample('sigmaepsilon_tform',
                                            dist.Uniform(0, (jnp.pi / 2.) * jnp.ones(N_knots_sig)))
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
            # eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_mag_batch(theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                      self.band_weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def initial_guess(self, reference_model="T21"):
        """

        Parameters
        ----------
        reference_model

        Returns
        -------

        """
        # Set hyperparameter initialisations
        param_root = f'model_files/{reference_model}_model'
        W0_init = np.loadtxt(f'{param_root}/W0.txt')
        l_knots = np.loadtxt(f'{param_root}/l_knots.txt')
        n_lknots, n_tauknots = W0_init.shape
        # W0_init = W0_init.flatten(order='F')
        W1_init = np.loadtxt(f'{param_root}/W1.txt')
        RV_init, tauA_init = np.loadtxt(f'{param_root}/M0_sigma0_RV_tauA.txt')[[2, 3]]

        # Interpolate to match new wavelength knots
        W0_init = interp1d(l_knots, W0_init, kind='cubic', axis=0)(self.l_knots)
        W1_init = interp1d(l_knots, W1_init, kind='cubic', axis=0)(self.l_knots)

        W0_init = W0_init.flatten(order='F')
        W1_init = W1_init.flatten(order='F')

        # I should remove all of this hardcoding
        n_eps = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
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
        param_init['Rv'] = jnp.array(3)
        param_init['tauA_tform'] = jnp.arctan(tauA_ / 1.)
        # param_init['tauA'] = tauA_
        param_init['sigma0_tform'] = jnp.arctan(sigma0_ / 0.1)
        param_init['sigma0'] = jnp.array(sigma0_)
        param_init['theta'] = jnp.array(np.random.normal(0, 1, n_sne))
        param_init['Av'] = jnp.array(np.random.exponential(tauA_, n_sne))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon_init), L_Omega_init)

        # param_init['theta'] = device_put(chains['theta'].mean(axis=(0, 1)))
        # param_init['Av'] = device_put(chains['AV'].mean(axis=(0, 1)))

        param_init['epsilon_tform'] = jnp.matmul(np.linalg.inv(L_Sigma), np.random.normal(0, 1, (n_eps, n_sne)))
        param_init['epsilon'] = np.random.normal(0, 1, (n_sne, n_eps))
        param_init['sigmaepsilon_tform'] = jnp.arctan(
            sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape) / 1.)
        param_init['sigmaepsilon'] = sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape)
        param_init['L_Omega'] = jnp.array(L_Omega_init)

        param_init['Ds'] = jnp.array(np.random.normal(self.data[-3, 0, :], sigma0_))

        return param_init

    def map_initial_guess(self):
        """

        Returns
        -------

        """
        optimizer = Adam(0.02)
        guide = AutoDelta(self.train_model)
        svi = SVI(self.train_model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(PRNGKey(123), 5000, self.data)
        params, losses = svi_result.params, svi_result.losses

        param_init = {}

        param_init['theta'] = params['theta_auto_loc']
        # sample_size = theta.shape[0]
        param_init['Av'] = params['AV_auto_loc']
        param_init['W0'] = params[
            'W0_auto_loc']  # .reshape((self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        param_init['W1'] = params[
            'W1_auto_loc']  # .reshape((self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        param_init['eps_tform'] = params['eps_tform_auto_loc']
        param_init['sigmaepsilon_tform'] = params['sigmaepsilon_tform_auto_loc']
        # sigmaepsilon = 1 * jnp.tan(sigmaepsilon_tform)
        param_init['L_Omega'] = params['L_Omega_auto_loc']
        # L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon), L_Omega)
        # eps = jnp.matmul(L_Sigma, eps_tform)
        # eps = eps.T
        # eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
        # eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
        # eps = eps_full.at[:, 1:-1, :].set(eps)
        param_init['Ds'] = params['Ds_auto_loc']
        param_init['Rv'] = params['Rv_auto_loc']

        print(param_init)
        raise ValueError('Nope')

        return param_init
        band_indices = self.data[-6, :, :].astype(int)
        mask = self.data[-1, :, :].astype(bool)

        test = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                   self.band_weights)
        for i in range(sample_size):
            plt.errorbar(self.data[0, :, i], self.data[1, :, i], yerr=self.data[2, :, i], fmt='x')
            plt.scatter(self.data[0, :, i], test[:, i])
            plt.show()

    def train(self, num_samples, num_warmup, num_chains, output, chain_method='parallel', init_strategy='median',
              mode='flux',
              l_knots=None):
        """

        Parameters
        ----------
        num_samples
        num_warmup
        num_chains
        output
        chain_method
        init_strategy
        mode
        l_knots

        Returns
        -------

        """
        if l_knots is not None:
            self.l_knots = jnp.array(l_knots)
            KD_l = spline_utils.invKD_irr(self.l_knots)
            self.J_l_T = device_put(spline_utils.spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        t = self.data[0, ...]
        self.hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])

        # -------------------------
        if init_strategy == 'value':
            init_strategy = init_to_value(values=self.initial_guess())
        elif init_strategy == 'map':
            init_strategy = init_to_value(values=self.map_initial_guess())
        elif init_strategy == 'median':
            init_strategy = init_to_median()
        elif init_strategy == 'sample':
            init_strategy = init_to_sample()
        else:
            raise ValueError('Invalid init strategy, must be one of value, median and sample')
        t = self.data[0, ...]
        self.hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        rng = PRNGKey(321)
        # rng = jnp.array([PRNGKey(11), PRNGKey(22), PRNGKey(33), PRNGKey(44)])
        # rng = PRNGKey(101)
        # numpyro.render_model(self.train_model, model_args=(self.data,), filename='train_model.pdf')
        nuts_kernel = NUTS(self.train_model, adapt_step_size=True, target_accept_prob=0.8, init_strategy=init_strategy,
                           dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False, step_size=10)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains,
                    chain_method=chain_method)
        mcmc.run(rng, self.data, extra_fields=('potential_energy',))
        mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        extras = mcmc.get_extra_fields(group_by_chain=True)
        self.train_postprocess(samples, extras, output)

    def train_postprocess(self, samples, extras, output):
        """

        Parameters
        ----------
        samples
        extras
        output

        Returns
        -------

        """
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

        # sigmaepsilon = np.mean(samples['sigmaepsilon'], axis=[0, 1])
        # L_Omega = np.mean(samples['L_Omega'], axis=[0, 1])
        # L_Sigma = np.matmul(np.diag(np.mean(samples['sigmaepsilon'], axis=[0, 1])), np.mean(samples['L_Omega'], axis=[0, 1]))
        sigma0 = np.mean(samples['sigma0'])

        Rv = np.mean(samples['Rv'])
        tauA = np.mean(samples['tauA'])
        M0_sigma0_RV_tauA = np.array([self.M0, sigma0, Rv, tauA])
        np.savetxt(os.path.join('results', output, 'W0.txt'), W0, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'W1.txt'), W1, delimiter="\t", fmt="%.3f")
        # np.savetxt(os.path.join('results', output, 'sigmaepsilon.txt'), sigmaepsilon, delimiter="\t", fmt="%.3f")
        # np.savetxt(os.path.join('results', output, 'L_Omega.txt'), L_Omega, delimiter="\t", fmt="%.3f")
        # np.savetxt(os.path.join('results', output, 'L_Sigma.txt'), L_Sigma, delimiter="\t", fmt="%.3f")
        np.savetxt(os.path.join('results', output, 'M0_sigma0_RV_tauA.txt'), M0_sigma0_RV_tauA, delimiter="\t",
                   fmt="%.3f")
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
        """

        Parameters
        ----------
        sample_name
        lc_dir
        meta_file
        map_dict
        sn_list
        data_mode

        Returns
        -------

        """
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
            data['mask'] = 1
            if data_mode == 'flux':
                lc = data[['t', 'flux', 'flux_err', 'band_indices', 'redshift', 'redshift_error', 'dist_mod', 'MWEBV',
                           'mask']]
                lc = lc.dropna(subset=['flux', 'flux_err'])
            else:
                lc = data[['t', 'MAG', 'MAGERR', 'band_indices', 'redshift', 'redshift_error', 'dist_mod', 'MWEBV',
                           'mask']]
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
        """

        Parameters
        ----------
        input_file

        Returns
        -------

        """
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
        mask = self.data[-1, ...]
        fit_mask = np.ones_like(fit_band_indices)
        model_flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, redshift, fit_band_indices, fit_mask)
        ts = np.linspace(-10, 40, 50)
        for _ in range(10):
            plt.figure()
            for i in range(4):
                inds = (band_indices[:, _] == i) & (mask[:, _] == 1)
                fit_inds = (fit_band_indices[:, _] == i) & (fit_mask[:, _] == 1)
                plt.errorbar(self.data[0, inds, _], self.data[1, inds, _], yerr=self.data[2, inds, _], fmt='x')
                plt.plot(ts, model_flux[fit_inds, _], ls='--')
        plt.show()

    def get_flux_from_chains(self, model):
        """

        Parameters
        ----------
        model

        Returns
        -------

        """
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
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1,
                                                                                                                     2,
                                                                                                                     0)
        J_t_hsiao = map(t, self.hsiao_t, self.KD_t_hsiao).reshape((*keep_shape, self.hsiao_t.shape[0]),
                                                                  order='F').transpose(1, 2, 0)
        t = t.reshape(keep_shape, order='F')
        band_indices = jnp.tile(np.array([[i] * 12 for i in range(4)]).flatten()[..., None], (1, sample_size))
        mask = jnp.ones_like(band_indices)

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
                       mask, J_t, J_t_hsiao)
        map = jax.vmap(jit_flux_batch, in_axes=(0, 0, None, None, 0, 0, None, None, None, None, None))
        flux = map(theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, mask, J_t, J_t_hsiao) / self.device_scale
        flux_bands = np.zeros((flux.shape[0], int(flux.shape[1] / steps_per_band), steps_per_band, flux.shape[-1]))
        for i in range(int(flux.shape[1] / steps_per_band)):
            flux_bands[:, i, ...] = flux[:, i * steps_per_band: (i + 1) * steps_per_band, ...]
        self.zp = np.array(
            [4.608419288004386e-09, 2.8305383925373084e-09, 1.917161265703195e-09, 1.446643295845274e-09])
        mag_bands = -2.5 * np.log10(flux_bands / self.zp[None, :, None, None])
        np.save(os.path.join('results', model, 'rf_mags'), mag_bands)

        eps = eps * 0

        flux = map(theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, mask, J_t, J_t_hsiao) / self.device_scale
        flux_bands = np.zeros((flux.shape[0], int(flux.shape[1] / steps_per_band), steps_per_band, flux.shape[-1]))
        for i in range(int(flux.shape[1] / steps_per_band)):
            flux_bands[:, i, ...] = flux[:, i * steps_per_band: (i + 1) * steps_per_band, ...]
        self.zp = np.array(
            [4.608419288004386e-09, 2.8305383925373084e-09, 1.917161265703195e-09, 1.446643295845274e-09])
        mag_bands = -2.5 * np.log10(flux_bands / self.zp[None, :, None, None])
        np.save(os.path.join('results', model, 'rf_mags_eps0'), mag_bands)

    def simulate_spectrum(self, t, N, dl=10, z=0, mu=0, ebv_mw=0, Rv=None, logM=None, del_M=None, AV=None, theta=None,
                          eps=None):
        """

        Parameters
        ----------
        t
        N
        dl
        z
        mu
        ebv_mw
        Rv
        logM
        del_M
        AV
        theta
        eps

        Returns
        -------

        """
        if del_M is None:
            del_M = self.sample_del_M(N)
        else:
            del_M = np.array(del_M)
            if len(del_M.shape) == 0:
                del_M = del_M.repeat(N)
            elif del_M.shape[0] != N:
                raise ValueError('If not providing a scalar del_M value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if AV is None:
            AV = self.sample_AV(N)
        else:
            AV = np.array(AV)
            if len(AV.shape) == 0:
                AV = AV.repeat(N)
            elif AV.shape[0] != N:
                raise ValueError('If not providing a scalar AV value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if theta is None:
            theta = self.sample_theta(N)
        else:
            theta = np.array(theta)
            if len(theta.shape) == 0:
                theta = theta.repeat(N)
            elif theta.shape[0] != N:
                raise ValueError('If not providing a scalar theta value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if eps is None:
            eps = self.sample_epsilon(N)
        elif len(np.array(eps).shape) == 0:
            eps = np.array(eps)
            if eps == 0:
                eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
            else:
                raise ValueError(
                    'For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots). The only scalar '
                    'value accepted is 0, which will effectively remove the effect of epsilon')
        elif len(eps.shape) != 3 or eps.shape[0] != N or eps.shape[1] != self.l_knots.shape[0] or eps.shape[2] != \
                self.tau_knots.shape[0]:
            raise ValueError('For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots)')
        ebv_mw = np.array(ebv_mw)
        if len(ebv_mw.shape) == 0:
            ebv_mw = ebv_mw.repeat(N)
        elif ebv_mw.shape[0] != N:
            raise ValueError(
                'For ebv_mw, either pass a single scalar value or an array of values for each of the N simulated objects')
        if Rv is None:
            Rv = self.Rv
        Rv = np.array(Rv)
        if len(Rv.shape) == 0:
            Rv = Rv.repeat(N)
        elif Rv.shape[0] != N:
            raise ValueError(
                'For Rv, either pass a single scalar value or an array of values for each of the N simulated objects')
        z = np.array(z)
        if len(z.shape) == 0:
            z = z.repeat(N)
        elif z.shape[0] != N:
            raise ValueError(
                'For z, either pass a single scalar value or an array of values for each of the N simulated objects')
        mu = np.array(mu)
        if len(mu.shape) == 0:
            mu = mu.repeat(N)
        elif mu.shape[0] != N:
            raise ValueError(
                'For mu, either pass a single scalar value or an array of values for each of the N simulated objects')
        param_dict = {
            'del_M': del_M,
            'AV': AV,
            'theta': theta,
            'eps': eps,
            'z': z,
            'mu': mu,
            'ebv_mw': ebv_mw,
            'Rv': Rv
        }
        l_r = np.linspace(min(self.l_knots), max(self.l_knots), int((max(self.l_knots) - min(self.l_knots)) / dl) + dl)
        l_o = l_r[None, ...].repeat(N, axis=0) * (1 + z[:, None])

        self.model_wave = l_r
        KD_l = spline_utils.invKD_irr(self.l_knots)
        self.J_l_T = device_put(spline_utils.spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        KD_x = spline_utils.invKD_irr(self.xk)
        self.M_fitz_block = device_put(spline_utils.spline_coeffs_irr(1e4 / self.model_wave, self.xk, KD_x))
        self.load_hsiao_template()

        t = jnp.array(t)
        t = jnp.repeat(t[..., None], N, axis=1)
        hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        keep_shape = t.shape
        t = t.flatten(order='F')
        map = jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None))
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1,
                                                                                                                     2,
                                                                                                                     0)
        spectra = self.get_spectra(theta, AV, self.W0, self.W1, eps, Rv, J_t, hsiao_interp)

        # Host extinction
        host_ext = np.zeros((N, l_r.shape[0], 1))
        for i in range(N):
            host_ext[i, :, 0] = extinction.fitzpatrick99(l_r, AV[i], Rv[i])

        # MW extinction
        mw_ext = np.zeros((N, l_o.shape[1], 1))
        for i in range(N):
            mw_ext[i, :, 0] = extinction.fitzpatrick99(l_o[i, ...], 3.1 * ebv_mw[i], 3.1)

        return l_o, spectra, param_dict

    def simulate_light_curve(self, t, N, bands, z=0, mu=0, ebv_mw=0, Rv=None, logM=None, del_M=None, AV=None,
                             theta=None, eps=None):
        """

        Parameters
        ----------
        t
        N
        bands
        z
        mu
        ebv_mw
        Rv
        logM
        del_M
        AV
        theta
        eps

        Returns
        -------

        """
        if del_M is None:
            del_M = self.sample_del_M(N)
        else:
            del_M = np.array(del_M)
            if len(del_M.shape) == 0:
                del_M = del_M.repeat(N)
            elif del_M.shape[0] != N:
                raise ValueError('If not providing a scalar del_M value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if AV is None:
            AV = self.sample_AV(N)
        else:
            AV = np.array(AV)
            if len(AV.shape) == 0:
                AV = AV.repeat(N)
            elif AV.shape[0] != N:
                raise ValueError('If not providing a scalar AV value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if theta is None:
            theta = self.sample_theta(N)
        else:
            theta = np.array(theta)
            if len(theta.shape) == 0:
                theta = theta.repeat(N)
            elif theta.shape[0] != N:
                raise ValueError('If not providing a scalar theta value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if eps is None:
            eps = self.sample_epsilon(N)
        elif len(np.array(eps).shape) == 0:
            eps = np.array(eps)
            if eps == 0:
                eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
            else:
                raise ValueError(
                    'For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots). The only scalar '
                    'value accepted is 0, which will effectively remove the effect of epsilon')
        elif len(eps.shape) != 3 or eps.shape[0] != N or eps.shape[1] != self.l_knots.shape[0] or eps.shape[2] != \
                self.tau_knots.shape[0]:
            raise ValueError('For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots)')
        ebv_mw = np.array(ebv_mw)
        if len(ebv_mw.shape) == 0:
            ebv_mw = ebv_mw.repeat(N)
        elif ebv_mw.shape[0] != N:
            raise ValueError(
                'For ebv_mw, either pass a single scalar value or an array of values for each of the N simulated objects')
        if Rv is None:
            Rv = self.Rv
        Rv = np.array(Rv)
        if len(Rv.shape) == 0:
            Rv = Rv.repeat(N)
        elif Rv.shape[0] != N:
            raise ValueError(
                'For Rv, either pass a single scalar value or an array of values for each of the N simulated objects')
        z = np.array(z)
        if len(z.shape) == 0:
            z = z.repeat(N)
        elif z.shape[0] != N:
            raise ValueError(
                'For z, either pass a single scalar value or an array of values for each of the N simulated objects')
        if mu == 'z':
            mu = self.cosmo.distmod(z).value
        else:
            mu = np.array(mu)
            if len(mu.shape) == 0:
                mu = mu.repeat(N)
            elif mu.shape[0] != N:
                raise ValueError(
                    'For mu, either pass a single scalar value or an array of values for each of the N simulated objects')
        param_dict = {
            'del_M': del_M,
            'AV': AV,
            'theta': theta,
            'eps': eps,
            'z': z,
            'mu': mu,
            'ebv_mw': ebv_mw,
            'Rv': Rv
        }

        t = jnp.array(t)
        num_per_band = t.shape[0]
        num_bands = len(bands)
        band_indices = np.zeros(num_bands * num_per_band)
        t = t[:, None].repeat(num_bands, axis=1).flatten(order='F')
        for i, band in enumerate(bands):
            if band not in self.band_dict.keys():
                raise ValueError(f'{band} is not included in current model')
            band_indices[i * num_per_band: (i + 1) * num_per_band] = self.band_dict[band]
        band_indices = band_indices[:, None].repeat(N, axis=1).astype(int)
        mask = np.ones_like(band_indices)
        band_weights = self._calculate_band_weights(z, ebv_mw)

        t = jnp.repeat(t[..., None], N, axis=1)
        hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        keep_shape = t.shape
        t = t.flatten(order='F')
        map = jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None))
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1,
                                                                                                                     2,
                                                                                                                     0)
        t = t.reshape(keep_shape, order='F')
        flux = self.get_mag_batch(theta, AV, self.W0, self.W1, eps, mu + del_M, Rv, band_indices, mask, J_t,
                                  hsiao_interp, band_weights)
        plt.scatter(t[:, 0], flux[:, 0])
        plt.show()

        return flux, param_dict

    def sample_del_M(self, N):
        """

        Parameters
        ----------
        N

        Returns
        -------

        """
        del_M = np.random.normal(0, self.sigma0, N)
        return del_M

    def sample_AV(self, N):
        """

        Parameters
        ----------
        N

        Returns
        -------

        """
        AV = np.random.exponential(self.tauA, N)
        return AV

    def sample_theta(self, N):
        """

        Parameters
        ----------
        N

        Returns
        -------

        """
        theta = np.random.normal(0, 1, N)
        return theta

    def sample_epsilon(self, N):
        """

        Parameters
        ----------
        N

        Returns
        -------

        """
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        eps_mu = jnp.zeros(N_knots_sig)
        eps = np.random.multivariate_normal(eps_mu, np.matmul(self.L_Sigma.T, self.L_Sigma), N)
        eps = np.reshape(eps, (N, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
        eps_full = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
        eps_full[:, 1:-1, :] = eps
        return eps_full

    def plot_hubble_diagram(self, model):
        """

        Parameters
        ----------
        model

        Returns
        -------

        """
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
