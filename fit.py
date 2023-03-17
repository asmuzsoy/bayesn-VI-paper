from bayesn_model import SEDmodel

model = SEDmodel()
filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
model.process_dataset('foundation', 'data/lcs/tables/T21_training_set.txt', 'data/lcs/meta/T21_training_set_meta.txt',
                      filt_map_dict, data_mode='mag')
model.fit(250, 250, 4, 'T21_fit_mag', chain_method='parallel')
