from bayesn_model import SEDmodel
import numpy as np

model = SEDmodel(load_model='T21_model')

dataset = 'sim_population'
# dataset = 'sim_nonzero_eps'

# dataset = 'T21_sim_2'

filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}

for i in range(100):
	print(i)

	sn_list = [int(i)]

	np.savetxt("temp_sn_list.txt", sn_list, fmt="%d", header="SNID", comments="")

	model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
	                      filt_map_dict, data_mode='flux', sn_list="temp_sn_list.txt")

	print("Fitting MCMC...")
	model.fit(250, 250, 4, "sim_population/" + str(i) + '_mcmc', chain_method='parallel', init_strategy='median')

	print("Fitting VI...")
	model.fit_with_vi("sim_population/" + str(i)  + '_vi', init_strategy='median')
