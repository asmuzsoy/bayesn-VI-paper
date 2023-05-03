import numpy as np
from timeit import default_timer
from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')
N = 100
start = default_timer()
lc, params = model.simulate_light_curve(np.arange(-8, 40, 4), N, ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1'],
                                        z=np.random.uniform(0, 0.1, N), mu='z', write_to_files=True, sim_name='T21_sim_100')
end = default_timer()
print(f'Simulating {N} objects took {end - start} seconds')
print(lc.shape)
