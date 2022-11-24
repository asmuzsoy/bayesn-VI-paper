import os
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, HMC
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

# jax.config.update('jax_platform_name', 'cpu')

print(jax.devices())

class Model(object):
    def __init__(self, bands, ignore_unknown_settings=False, settings={}, device='cuda'):
        self.data = None
        self.device = device
        self.settings = parse_settings(bands, settings,
                                       ignore_unknown_settings=ignore_unknown_settings)
        self.M0 = -19.5

        self.l_knots = np.genfromtxt('model_files/T21_model/l_knots.txt')
        self.tau_knots = np.genfromtxt('model_files/T21_model/tau_knots.txt')
        self.W0 = np.genfromtxt('model_files/T21_model/W0.txt')
        self.W1 = np.genfromtxt('model_files/T21_model/W1.txt')
        self.l_knots = device_put(self.l_knots)
        self.tau_knots = device_put(self.tau_knots)
        self.W0 = device_put(self.W0)
        self.W1 = device_put(self.W1)
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

    def get_flux_batch(self, theta, Av, redshifts, band_indices):
        num_batch = theta.shape[0]
        J_t = jnp.reshape(self.J_t, (-1, *self.J_t.shape))
        J_t = jnp.repeat(J_t, num_batch, axis=0)
        J_t_hsiao = jnp.reshape(self.J_t_hsiao, (-1, *self.J_t_hsiao.shape))
        J_t_hsiao = jnp.repeat(J_t_hsiao, num_batch, axis=0)
        W0 = jnp.reshape(self.W0, (-1, *self.W0.shape))
        W0 = jnp.repeat(W0, num_batch, axis=0)
        W1 = jnp.reshape(self.W1, (-1, *self.W1.shape))
        W1 = jnp.repeat(W1, num_batch, axis=0)

        W = W0 + theta[..., None, None] * W1

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

        model_flux = model_flux * 10 ** -(0.4 * self.M0)

        return model_flux

    def model(self, obs):
        sample_size = self.data.shape[-1]
        # for sn_index in pyro.plate('SNe', sample_size):
        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            # Rv = numpyro.sample(f'RV', dist.Normal(2.610, 0.001))
            Av = numpyro.sample(f'AV', dist.Exponential(0.194))
            band_indices = obs[-2, :, sn_index].astype(int).T
            redshift = obs[-1, 0, sn_index]
            start = time.time()
            flux = self.get_flux_batch(theta, Av, redshift, band_indices)
            end = time.time()
            elapsed = end - start
            self.integ_time += elapsed
            self.total += sample_size
            """if self.count > -1:
                for i in range(4):
                    inds = band_indices[:, 0]
                    inds = inds == i
                    print(t.shape)
                    print(obs.shape)
                    print(flux.shape)
                    # plt.scatter(t[inds, :], flux.detach().numpy()[inds, :])
                    # plt.errorbar(t[inds, 0], torch.squeeze(obs[1, inds, :]), yerr=torch.squeeze(obs[2, inds, :]), fmt='x')
                    plt.scatter(t[inds, 0], flux.detach().numpy()[inds, 0])
                    plt.errorbar(t[inds, 0], obs[1, inds, 0], yerr=obs[2, inds, 0], fmt='x')
                    plt.show()
                    raise ValueError('Nope')
                plt.title(theta.detach().numpy())
                plt.show()
                raise ValueError('Nope')"""
            self.count += 1
            numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T) # _{sn_index}

    def fit(self, dataset):
        self.process_dataset(dataset)
        self.integ_time = 0
        self.total = 0
        self.count = 0
        self.thetas = []
        rng = PRNGKey(123)
        # pyro.render_model(self.model, model_args=(self.data,), filename='model.pdf')
        nuts_kernel = NUTS(self.model, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=250, num_warmup=250, num_chains=1)
        mcmc.run(rng, self.data)  # self.rng,
        print(f'{self.total * self.data.shape[1]} flux integrals for {self.total} objects in {self.integ_time} seconds')
        print(f'Average per object: {self.integ_time / self.total}')
        print(f'Average per integral: {self.integ_time / (self.total * self.data.shape[1])}')
        print(np.array(self.thetas))
        return mcmc.get_samples()

    def process_dataset(self, dataset):
        all_data = []
        self.t = None
        for lc in dataset.light_curves:
            lc = lc.to_pandas()
            lc = lc.astype({'band': str})
            lc[['flux', 'fluxerr']] = lc[['flux', 'fluxerr']] # / self.scale
            lc['band'] = lc['band'].apply(lambda band: band[band.find("'") + 1: band.rfind("'")])
            lc['band'] = lc['band'].apply(lambda band: self.band_dict[band])
            lc['redshift'] = 0
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
    dataset_path = 'data/bayesn_sim_test_z0_ext_25000.h5'
    dataset = lcdata.read_hdf5(dataset_path)[:100]
    bands = set()
    for lc in dataset.light_curves:
        bands = bands.union(lc['band'])
    bands = np.array(sorted(bands, key=get_band_effective_wavelength))

    param_path = 'data/bayesn_sim_test_z0_ext_25000_params.pkl'
    params = pickle.load(open(param_path, 'rb'))
    del params['epsilon']
    params = pd.DataFrame(params)

    pd_dataset = dataset.meta.to_pandas()
    pd_dataset = pd_dataset.astype({'object_id': int})
    params = pd_dataset.merge(params, on='object_id')

    model = Model(bands, device='cuda')
    # model.compare_gen_theta(dataset, params)
    result = model.fit(dataset)
    for p in result.keys():
        params[f'fit_mu_{p}'] = np.mean(result[p], axis=0)
        params[f'fit_sigma_{p}'] = np.std(result[p], axis=0)
        print(params[[p, f'fit_mu_{p}', f'fit_sigma_{p}']])
    # print(np.mean(result['theta_1'].numpy(), axis=0), np.std(result['theta_1'].numpy(), axis=0))
    # plt.scatter(params.theta.values, result['theta'][0, :])
    # plt.show()



