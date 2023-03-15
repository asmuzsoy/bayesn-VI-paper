import numpy as np
from timeit import default_timer
from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')
N = 10000
lc, params = model.simulate_light_curve(np.arange(-8, 40, 4), N, ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1'],
                                        z=np.random.uniform(0, 0.1, N), mu='z', sim_name='T21_sim_10000')
