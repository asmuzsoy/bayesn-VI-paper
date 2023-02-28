from model import Model

model = Model()
filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
model.process_dataset('foundation', 'data/LCs/foundation/Foundation_DR1', 'data/LCs/meta/T21_training_set_meta.txt',
                      filt_map_dict, sn_list='data/LCs/foundation/Foundation_DR1/Foundation_DR1.LIST')
model.fit(250, 250, 4, 'foundation_fit_noeps_test', 'T21', chain_method='vectorized')
