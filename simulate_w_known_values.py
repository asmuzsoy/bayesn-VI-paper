import numpy as np
from timeit import default_timer
from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')


N = 1
start = default_timer()
print(np.arange(-8, 40, 4))

lc, params = model.simulate_light_curve(np.arange(-8, 40, 4), N, ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1'],
										AV = 0.0, theta = 2.,
                                        z=0.02, mu='z', write_to_files=True, sim_name='sim_zero_AV')
end = default_timer()
print(f'Simulating {N} objects took {end - start} seconds')
print(lc.shape)
