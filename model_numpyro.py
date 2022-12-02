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

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})

# jax.config.update('jax_platform_name', 'cpu')
# numpyro.set_host_device_count(4)

print(jax.devices())


class Model(object):
    def __init__(self, bands, ignore_unknown_settings=False, settings={}, device='cuda',
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}):
        self.data = None
        self.device = device
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
        """self.W0 = np.array([[-0.09999619, 0.23204613, 0.18652959, 0.28380015,
                             0.36287796, 2.0998864],
                            [0.14592355, 0.35849518, 0.32490727, 0.41864118,
                             0.46873084, 1.1563009],
                            [0.34741428, 0.39980915, 0.40663296, 0.43617624,
                             0.42968237, 0.26654527],
                            [0.3710753, 0.4344279, 0.36928922, 0.37977916,
                             0.5314234, 0.69516015],
                            [-0.27417222, 0.5177755, 0.41066182, 0.52734554,
                             0.5681232, 0.78215384],
                            [0.27546242, 0.32729372, 0.31196332, 0.39779627,
                             0.47519293, 0.37028623]])
        self.W1 = np.array([[-0.3455001, -0.04963814, -0.63791806, 0.0071786,
                             -0.45937502, 0.18831591],
                            [-0.34769145, -0.17091192, -0.193374, -0.41438517,
                             -0.35414574, -0.3171949],
                            [-0.25707978, -0.13066787, -0.18412727, -0.10917296,
                             -0.38949892, -0.2089294],
                            [-0.24793991, -0.06219663, -0.11003418, 0.01114025,
                             -0.29314718, -0.28556004],
                            [-0.18288808, -0.16517894, -0.10562517, -0.09177864,
                             -0.32245842, -0.457084],
                            [0.1985342, -0.3601357, 0.23641136, -0.82539576,
                             -0.79625946, -1.0284847]])
        self.L_Sigma = np.array([[1.43638805e-01, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [8.56567100e-02, 1.65818870e-01, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [1.36759713e-01, 8.02962035e-02, 1.79056808e-01,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [1.01001579e-02, 2.87260972e-02, 2.14673914e-02,
                                  1.25275671e-01, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [2.18518195e-03, 1.34234931e-02, 7.23335007e-03,
                                  4.64537041e-03, 5.50385825e-02, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-8.97977781e-03, -1.60565116e-02, -6.53153425e-03,
                                  1.05541181e-02, 2.01088265e-02, 4.19724844e-02,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-8.78980570e-03, -2.80204671e-03, -2.21497864e-02,
                                  -7.57818623e-03, 3.72570567e-03, -1.27985217e-02,
                                  5.21318391e-02, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-3.73327080e-03, -1.25274872e-02, 3.79547453e-03,
                                  -1.72037221e-02, 8.60234816e-03, 8.41650460e-03,
                                  1.06696477e-02, 5.54007962e-02, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [1.55449985e-02, -1.76404864e-02, 5.55917912e-04,
                                  1.09584955e-02, 9.29977465e-03, 1.08678006e-02,
                                  -1.21023208e-02, 1.01483995e-02, 5.45607880e-02,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [1.27102991e-04, 8.02196935e-03, 2.21286272e-03,
                                  -1.29162595e-02, -1.80717688e-02, 7.34244892e-03,
                                  -2.72246804e-02, -2.31613815e-02, -8.01646058e-03,
                                  6.19728602e-02, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [8.85064527e-03, 1.30197518e-02, 2.77073551e-02,
                                  -2.84573007e-02, -4.17077616e-02, -1.18092438e-02,
                                  -2.39622239e-02, -2.82625612e-02, 1.42890215e-02,
                                  9.97292530e-03, 7.40535483e-02, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [5.00759576e-03, 2.16773339e-02, 3.78916389e-03,
                                  -1.51035683e-02, -3.01679708e-02, -6.14898512e-03,
                                  -1.64471474e-02, 3.35187581e-03, -7.52407068e-04,
                                  1.48020079e-02, 2.28462480e-02, 5.39939702e-02,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-4.89116088e-03, -9.62143298e-03, -2.14465149e-02,
                                  8.73274822e-03, 2.36150622e-02, 5.17971208e-03,
                                  -1.07753845e-02, 1.84248500e-02, -1.19999452e-02,
                                  -8.78675934e-03, -7.19203567e-03, -8.94674193e-03,
                                  7.05073103e-02, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-7.69580423e-04, -3.09699606e-02, -1.15795424e-02,
                                  -1.74971167e-02, -1.27174947e-02, 1.07036866e-02,
                                  -1.27562769e-02, -2.35657804e-02, 1.21224085e-02,
                                  4.71792463e-03, -3.01985652e-03, -4.16063471e-03,
                                  -1.60764046e-02, 5.51190190e-02, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [1.13138026e-02, 4.15176665e-03, -1.31124556e-02,
                                  -3.25250141e-02, -1.38252750e-02, 7.09223095e-03,
                                  5.07954601e-03, -2.68539991e-02, -2.33370382e-02,
                                  -1.01598026e-02, 2.05492526e-02, 2.88733374e-02,
                                  -2.88612135e-02, 2.77828444e-02, 8.04149434e-02,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [7.41722703e-04, 3.15276184e-03, -3.15486337e-04,
                                  -2.97203325e-02, 2.12194328e-03, -3.50042037e-03,
                                  -9.10992175e-03, 1.04691461e-02, -1.33785896e-03,
                                  -4.86506009e-03, 1.71241281e-03, 1.08834822e-02,
                                  -1.38235884e-02, 3.84188183e-02, 2.76670028e-02,
                                  3.22814621e-02, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-8.46088442e-05, 7.31532136e-03, 1.58400256e-02,
                                  6.56705862e-03, 1.42485127e-02, -4.04518144e-03,
                                  1.62074354e-03, -1.97375212e-02, -7.12730456e-03,
                                  1.71015598e-02, 2.44513806e-02, -1.13668162e-02,
                                  1.83782633e-02, -2.16764696e-02, 1.31115830e-02,
                                  1.76070072e-02, 8.23379382e-02, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-1.30223008e-02, -1.24558192e-02, 1.19952601e-03,
                                  -1.81919677e-04, -7.08526140e-03, 3.56124947e-03,
                                  -1.21705662e-02, -1.97273903e-02, 2.87793693e-03,
                                  2.55662967e-02, -9.45861265e-03, -3.91059992e-04,
                                  5.62069193e-03, 1.22120120e-02, -5.95443125e-04,
                                  -5.55024110e-03, -1.07686128e-02, 4.83113155e-02,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-7.42948847e-03, -8.59929994e-03, -3.30923242e-03,
                                  -7.16110319e-03, -2.96753608e-02, 7.61826814e-04,
                                  7.60360761e-03, -1.06761567e-02, 3.47070931e-03,
                                  -1.89481564e-02, 8.70919507e-03, -7.16275955e-03,
                                  -9.09406692e-03, 1.39581747e-02, 9.62820556e-03,
                                  -1.42305240e-03, -3.76847386e-02, 1.90171283e-02,
                                  5.42742498e-02, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-1.56931430e-02, 1.29073709e-02, 4.98498650e-03,
                                  -1.65603925e-02, -1.55064724e-02, -8.03397223e-03,
                                  6.75263116e-03, 2.60919915e-03, -8.21535103e-03,
                                  -2.27692183e-02, -2.41619069e-03, 3.09554227e-02,
                                  -1.76761746e-02, 3.53050753e-02, 1.22870656e-03,
                                  9.05176811e-03, -2.38256827e-02, 1.51671320e-02,
                                  2.22142506e-02, 3.76693420e-02, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-2.15921197e-02, 1.28978007e-02, -3.25418003e-02,
                                  -7.14274414e-04, 3.23209204e-02, -2.33958717e-02,
                                  4.73687775e-04, 5.75235412e-02, -4.82030064e-02,
                                  -4.45731170e-02, 1.24657722e-02, 2.04063877e-02,
                                  4.94075976e-02, 1.60207991e-02, 1.41093479e-02,
                                  -2.74532586e-02, 4.24093654e-04, -8.25113989e-03,
                                  -1.72103718e-02, -3.50753330e-02, 7.72410110e-02,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [5.14790445e-05, 4.74491622e-03, 2.63115275e-03,
                                  -1.47063993e-02, -1.34488400e-02, 6.46355841e-03,
                                  1.42638572e-03, -3.34873870e-02, -2.17428356e-02,
                                  3.14250179e-02, 1.33808516e-02, -4.82461601e-03,
                                  -1.69698577e-02, -1.74638443e-03, -1.11873215e-03,
                                  1.37232607e-02, 8.27789679e-03, -4.25552316e-02,
                                  -6.17485552e-04, -4.36041467e-02, -1.28847100e-02,
                                  6.71188682e-02, 0.00000000e+00, 0.00000000e+00],
                                 [-2.40780204e-03, 2.69217379e-02, -1.75824128e-02,
                                  -1.22304345e-02, -1.69690289e-02, 1.37429184e-03,
                                  -2.72568013e-03, -4.82610427e-02, -3.64305414e-02,
                                  -3.22555914e-03, 2.26743948e-02, 4.70765904e-02,
                                  3.30856368e-02, 8.10475647e-03, 2.03615017e-02,
                                  4.81473980e-03, 4.02321331e-02, -2.58045085e-03,
                                  -2.90889926e-02, -1.67682637e-02, 4.54396242e-03,
                                  4.57852520e-02, 5.20353243e-02, 0.00000000e+00],
                                 [1.49633950e-02, 2.65404694e-02, -3.92465033e-02,
                                  -1.74919497e-02, -3.39837819e-02, -8.38994794e-03,
                                  4.04178863e-03, -4.09166561e-03, -9.38975811e-03,
                                  -1.47241522e-02, 3.85021269e-02, -1.18346401e-02,
                                  -1.82129815e-02, 4.49067447e-03, -2.23424938e-02,
                                  1.98660847e-02, -3.41089279e-03, -2.33754143e-02,
                                  7.62001891e-03, -1.14859883e-02, 1.31796673e-02,
                                  6.34849211e-03, 2.59032566e-02, 4.51683067e-02]])"""

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
        sigmaepsilon = numpyro.sample('sigmaepsilon', dist.HalfNormal(1 * jnp.ones(N_knots_sig)))
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
            flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, redshift, band_indices)

            end = time.time()
            elapsed = end - start
            numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)  # _{sn_index}

    def train(self, dataset):
        self.process_dataset(dataset)
        rng = PRNGKey(123)
        # numpyro.render_model(self.train_model, model_args=(self.data,), filename='train_model.pdf')
        nuts_kernel = NUTS(self.train_model, adapt_step_size=True, target_accept_prob=0.9, init_strategy=init_to_median())
        mcmc = MCMC(nuts_kernel, num_samples=250, num_warmup=250, num_chains=1)
        mcmc.run(rng, self.data)
        return mcmc

    def train_assess(self, params, yaml_dir):
        with open(os.path.join('results', f'{yaml_dir}.yaml'), 'r') as file:
            result = yaml.load(file, yaml.Loader)
        print(result['theta'].shape)
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

    def process_dataset(self, dataset):
        all_data = []
        self.t = None
        for lc_ind, lc in enumerate(dataset.light_curves):
            lc = lc.to_pandas()
            lc = lc.astype({'band': str})
            lc[['flux', 'fluxerr']] = lc[['flux', 'fluxerr']]  * self.scale
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
    dataset_path = 'data/bayesn_sim_team_z0.1_daily_25000.h5'
    dataset = lcdata.read_hdf5(dataset_path)[:1000]
    bands = set()
    for lc in dataset.light_curves:
        bands = bands.union(lc['band'])
    bands = np.array(sorted(bands, key=get_band_effective_wavelength))

    param_path = 'data/bayesn_sim_team_z0.1_daily_25000_params.csv'
    params = pd.read_csv(param_path)

    pd_dataset = dataset.meta.to_pandas()
    pd_dataset = pd_dataset.astype({'object_id': int})
    params = pd_dataset.merge(params, on='object_id')

    model = Model(bands, device='cuda')
    # result = model.fit(dataset)
    # result = model.train(dataset)
    # result.print_summary()
    # model.save_results_to_yaml(result, 'daily_cadence_train')
    # model.fit_assess(params, '4chain_fit_test')
    # model.fit_from_results(dataset, 'gpu_train_dist')
    model.train_assess(params, '4chain_train_test')
