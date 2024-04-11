import numpy as np
from timeit import default_timer
from bayesn_model import SEDmodel
import os.path

model = SEDmodel(load_model='T21_model')


dataset_number = 30


epsilons_on = True

# if os.path.exists("sim_population_AV_" + str(dataset_number) + ".txt"):
#     raise ValueError("It looks like a dataset with this name already exists.")


N = 500

start = default_timer()

# AV = np.zeros(N)
AV = model.sample_AV(N)
# AV = np.random.uniform(1e-9, 0.25, N)
# AV = np.random.exponential(0.1, N)

z = np.random.uniform(0.015, 0.08, N)
theta = model.sample_theta(N)

if epsilons_on:
    lc, params = model.simulate_light_curve(np.arange(-8, 40, 4), N, ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1'], 
    									AV = AV, theta = theta, 
                                        z=z, mu='z', write_to_files=True, 
                                        sim_name='sim_population_' + str(dataset_number))
else:
    lc, params = model.simulate_light_curve(np.arange(-8, 40, 4), N, ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1'], 
                                    AV = AV, theta = theta, 
                                    z=z, mu='z', write_to_files=True, 
                                    sim_name='sim_population_' + str(dataset_number), eps=0)

np.savetxt("sim_population_AV_" + str(dataset_number) + ".txt", AV)
np.savetxt("sim_population_theta_" + str(dataset_number) + ".txt", theta)
np.savetxt("sim_population_z_" + str(dataset_number) + ".txt", z)


end = default_timer()
print(f'Simulating {N} objects took {end - start} seconds')
print(lc.shape)
