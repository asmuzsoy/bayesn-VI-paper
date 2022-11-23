import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pyro
from pyro.infer import MCMC, NUTS, Predictive
import pyro.distributions as dist
import h5py
import lcdata
import sncosmo
from settings import parse_settings
import spline_utils
import time

class Model(object):
    def __init__(self, bands, ignore_unknown_settings=False, settings={}, device='cuda'):
        self.data = None
        self.device = device
        self.settings = parse_settings(bands, settings,
                                       ignore_unknown_settings=ignore_unknown_settings)
        self.M0 = -19.5

        self.l_knots = np.genfromtxt('model_files/T21_model/l_knots.txt')
        self.tau_knots = np.genfromtxt('model_files/T21_model/tau_knots.txt')
        self.W0 = torch.from_numpy(np.genfromtxt('model_files/T21_model/W0.txt'))
        self.W1 = torch.from_numpy(np.genfromtxt('model_files/T21_model/W1.txt'))
        self.band_dict = {band: i for i, band in enumerate(bands)}

        self._setup_band_weights()

        KD_l = spline_utils.invKD_irr(self.l_knots)
        self.J_l_T = torch.from_numpy(spline_utils.spline_coeffs_irr(self.model_wave, self.l_knots, KD_l)).float()
        self.KD_t = spline_utils.invKD_irr(self.tau_knots)
        self.load_hsiao_template()
        self.scale = 1e18

        self.W0 = self.W0.to(self.device)
        self.W1 = self.W1.to(self.device)
        self.ZPT = 27.5
        self.J_l_T = self.J_l_T.to(self.device)
        self.hsiao_flux = self.hsiao_flux.to(self.device)
        self.J_l_T_hsiao = self.J_l_T_hsiao.to(self.device)

    def load_hsiao_template(self):
        with h5py.File(os.path.join('data', 'hsiao.h5'), 'r') as file:
            data = file['default']

            hsiao_phase = data['phase'][()].astype('float64')
            hsiao_wave = data['wave'][()].astype('float64')
            hsiao_flux = data['flux'][()].astype('float64')

        KD_l_hsiao = spline_utils.invKD_irr(hsiao_wave)
        self.KD_t_hsiao = spline_utils.invKD_irr(hsiao_phase)
        self.J_l_T_hsiao = torch.from_numpy(spline_utils.spline_coeffs_irr(self.model_wave,
                                                                           hsiao_wave, KD_l_hsiao)).float()

        self.hsiao_t = hsiao_phase
        self.hsiao_l = hsiao_wave
        self.hsiao_flux = torch.from_numpy(hsiao_flux).T.float()

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
            band_conv_transmission = np.convolve(
                band_transmission * band_pad_dwave,
                np.ones(self.settings['band_oversampling']),
                mode='valid'
            )
            band_conv_transmission = np.interp(band_wave, 10 ** band_pad_log_wave, band_transmission)

            band_weight = (
                    band_wave
                    * band_conv_transmission
                    / sncosmo.constants.HC_ERG_AA
                    / ref.zpbandflux(band)
                    * 10 ** (0.4 * -20.)
            )

            dlamba = np.diff(band_wave)
            dlamba = np.r_[dlamba, dlamba[-1]]

            num = band_wave * band_conv_transmission * dlamba
            denom = np.sum(num)
            band_weight = num / denom

            band_weights.append(band_weight)

        # Get the locations that should be sampled at redshift 0. We can scale these to
        # get the locations at any redshift.
        band_interpolate_locations = torch.arange(
            0,
            self.settings['spectrum_bins'] * self.settings['band_oversampling'],
            self.settings['band_oversampling']
        )

        # Save the variables that we need to do interpolation.
        self.band_interpolate_locations = band_interpolate_locations.to(self.device)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = torch.FloatTensor(band_weights).to(self.device)
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
                + torch.log10(1 + redshifts)[:, None] / self.band_interpolate_spacing
        )

        flat_locs = locs.flatten()

        # Linear interpolation
        int_locs = flat_locs.long()
        remainders = flat_locs - int_locs

        start = self.band_interpolate_weights[..., int_locs]
        end = self.band_interpolate_weights[..., int_locs + 1]

        flat_result = remainders * end + (1 - remainders) * start
        result = flat_result.reshape((-1,) + locs.shape).permute(1, 2, 0)

        # We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        # Hack fix maybe
        sum = torch.sum(result, axis=1)
        result /= sum[:, None, :]

        return result

    def get_flux(self, theta, t, redshifts, band_indices):
        theta = 0 * theta - 3
        J_t = torch.from_numpy(spline_utils.spline_coeffs_irr(t, self.tau_knots, self.KD_t).T)
        J_t_hsiao = torch.from_numpy(spline_utils.spline_coeffs_irr(t, self.hsiao_t,
                                                                              self.KD_t_hsiao).T)
        J_t = J_t.to(self.device)
        J_t = J_t.float()
        J_t_hsiao = J_t_hsiao.to(self.device)
        J_t_hsiao = J_t_hsiao.float()
        # W0 = torch.reshape(self.W0, (-1, *self.W0.shape))
        # W0 = torch.repeat_interleave(W0, num_batch, dim=0)
        # W1 = torch.reshape(self.W1, (-1, *self.W1.shape))
        # W1 = torch.repeat_interleave(W1, num_batch, dim=0)

        redshifts = redshifts[None]

        W = self.W0 + theta[..., None] * self.W1
        W = W.float()

        WJt = torch.matmul(W, J_t)
        W_grid = torch.matmul(self.J_l_T, WJt)

        HJt = torch.matmul(self.hsiao_flux, J_t_hsiao)
        H_grid = torch.matmul(self.J_l_T_hsiao, HJt)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)

        num_observations = t.shape[0]

        band_weights = self._calculate_band_weights(redshifts)

        obs_band_weights = (
            band_weights[0, :, band_indices.flatten()]
        )

        model_flux = torch.sum(model_spectra * obs_band_weights, axis=0)

        # Out by factor of 10^8 for some reason, hack fix but investigate this
        model_flux = model_flux * 10 ** (-0.4 * self.M0)
        print(model_flux)
        raise ValueError('Nope')

        return model_flux

    def get_flux_batch(self, theta, t, redshifts, band_indices):
        num_batch = theta.shape[0]
        J_t_list, J_t_hsiao_list = [], []
        for _ in range(num_batch):
            J_t_list.append(torch.from_numpy(spline_utils.spline_coeffs_irr(t[:, _], self.tau_knots, self.KD_t).T))
            J_t_hsiao_list.append(torch.from_numpy(spline_utils.spline_coeffs_irr(t[:, _], self.hsiao_t,
                                                                                  self.KD_t_hsiao).T))
        J_t, J_t_hsiao = torch.stack(J_t_list), torch.stack(J_t_hsiao_list)
        J_t = J_t.to(self.device)
        J_t = J_t.float()
        J_t_hsiao = J_t_hsiao.to(self.device)
        J_t_hsiao = J_t_hsiao.float()
        W0 = torch.reshape(self.W0, (-1, *self.W0.shape))
        W0 = torch.repeat_interleave(W0, num_batch, dim=0)
        W1 = torch.reshape(self.W1, (-1, *self.W1.shape))
        W1 = torch.repeat_interleave(W1, num_batch, dim=0)

        W = W0 + theta[..., None, None] * W1
        W = W.float()

        WJt = torch.matmul(W, J_t)
        W_grid = torch.matmul(self.J_l_T, WJt)

        HJt = torch.matmul(self.hsiao_flux, J_t_hsiao)
        H_grid = torch.matmul(self.J_l_T_hsiao, HJt)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)
        """
        integ1 = self.model_wave[:, None] * model_spectra_np * band[:, None]
        integ2 = self.model_wave * band
        ref = sncosmo.get_magsystem(self.settings['magsys'])
        f1 = np.array([np.trapz(integ1[:, i], x=self.model_wave) / np.trapz(integ2, x=self.model_wave) for i in range(integ1.shape[-1])])
        m1 = -2.5 * np.log10(f1 / zp)
        print(m1)
        zp = 6.32e-9
        dlamba = np.diff(self.model_wave)
        dlamba = np.r_[dlamba, dlamba[-1]]
        num = self.model_wave * band * dlamba
        denom = np.sum(num)
        weight = num / denom
        inds = band_indices[:, 0] == 1
        f1 = np.array([np.sum(model_spectra_np[:, i] * weight) for i in range(model_spectra_np.shape[-1])])[inds]
        m1 = -2.5 * np.log10(f1 / zp)"""

        num_observations = t.shape[0]

        band_weights = self._calculate_band_weights(redshifts)
        batch_indices = (
            torch.arange(num_batch, device=self.device)
            .repeat_interleave(num_observations)
        )

        obs_band_weights = (
            band_weights[batch_indices, :, band_indices.T.flatten()]
            .reshape((num_batch, num_observations, -1))
            .permute(0, 2, 1)
        )

        model_flux = torch.sum(model_spectra * obs_band_weights, axis=1).T

        model_flux = model_flux * 10 ** -(0.4 * self.M0)
        #f2 = model_flux[inds, 0]
        #m2 = -2.5 * np.log10(f2 / zp)
        #m2 = m2.detach().numpy()
        #print(f1 / f2)
        #print(m2)
        #print(m1 - m2)
        return model_flux

    def model(self, obs):
        sample_size = self.data.shape[-1]
        #for sn_index in pyro.plate('SNe', sample_size):
        with pyro.plate('SNe', sample_size) as sn_index:
            theta = pyro.sample(f'theta', dist.Normal(0, torch.tensor(1.0, device=self.device))) # _{sn_index}
            t = obs[0, :, sn_index].cpu().numpy()
            band_indices = obs[-2, :, sn_index].long()
            redshift = obs[-1, 0, sn_index]
            start = time.time()
            flux = self.get_flux_batch(theta, t, redshift, band_indices)
            end = time.time()
            elapsed = end - start
            self.integ_time += elapsed
            self.total += sample_size
            self.thetas.append(theta.detach().numpy())
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
            pyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index]), obs=obs[1, :, sn_index]) # _{sn_index}
            #print('-----------')
            #print(flux)
            #print(obs[2, :, sn_index])
            #print(torch.normal(flux, obs[2, :, sn_index]))
            #print(type(x))

    def fit(self, dataset):
        self.process_dataset(dataset)
        self.integ_time = 0
        self.total = 0
        self.count = 0
        self.thetas = []
        # pyro.render_model(self.model, model_args=(self.data,), filename='model.pdf')
        nuts_kernel = NUTS(self.model, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=50, warmup_steps=500, num_chains=1)
        mcmc.run(self.data)  # self.rng,
        print(f'{self.total * self.data.shape[1]} flux integrals for {self.total} objects in {self.integ_time} seconds')
        print(f'Average per object: {self.integ_time / self.total}')
        print(f'Average per integral: {self.integ_time / (self.total * self.data.shape[0])}')
        print(np.array(self.thetas))
        return mcmc.get_samples()

    def process_dataset(self, dataset):
        all_data = []
        for lc in dataset.light_curves:
            lc = lc.to_pandas()
            lc = lc.astype({'band': str})
            lc[['flux', 'fluxerr']] = lc[['flux', 'fluxerr']] # / self.scale
            lc['band'] = lc['band'].apply(lambda band: band[band.find("'") + 1: band.rfind("'")])
            lc['band'] = lc['band'].apply(lambda band: self.band_dict[band])
            lc['redshift'] = 0
            lc = lc.sort_values('time')
            lc = lc.values
            lc = np.reshape(lc, (-1, *lc.shape))
            all_data.append(lc)
        all_data = np.concatenate(all_data)
        all_data = np.swapaxes(all_data, 0, 2)
        all_data = torch.from_numpy(all_data)
        self.data = all_data.to(self.device)

    def compare_gen_theta(self, dataset, params):
        self.process_dataset(dataset)
        for ind in range(self.data.shape[-1]):
            lc = self.data[..., ind]
            lc = torch.reshape(lc, (*lc.shape, -1))
            t, band_indices, redshifts = lc[0, ...].cpu().numpy(), lc[-2, ...].long(), lc[-1, 0, :]
            theta = params['theta'].values[ind]
            print(theta)
            theta = torch.tensor(theta)
            theta = torch.reshape(theta, (*theta.shape, -1))
            fl = self.get_flux_batch(theta, t, redshifts, band_indices).cpu().numpy()
            band_indices = band_indices.cpu().numpy()
            lc = lc[..., 0].cpu().numpy()
            t, fl, band_indices = np.squeeze(t), np.squeeze(fl), np.squeeze(band_indices)
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            for i in range(4):
                a = ax.flatten()[i]
                inds = band_indices == i
                a.scatter(t[inds], fl[inds])
                a.errorbar(t[inds], lc[1, inds], yerr=lc[2, inds], fmt='x')
            plt.suptitle(theta)
            plt.show()




