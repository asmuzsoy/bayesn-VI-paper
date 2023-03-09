from bayesn_model import SEDmodel

model = SEDmodel()
filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
model.process_dataset('foundation', 'data/LCs/Foundation_DR1/Foundation_DR1', 'data/LCs/meta/T21_training_set_meta.txt',
                      filt_map_dict, sn_list='data/LCs/Foundation_DR1/Foundation_DR1/Foundation_DR1.LIST', data_mode='mag')
#model.process_dataset('ztf', 'data/LCs/ZTF', 'data/LCs/meta/ztf_dr1_training.txt',
#                      map_dict=None, data_mode='mag')
model.train(20, 20, 4, 'quick_test', chain_method='parallel', init_strategy='value')
#            l_knots=[4150, 4760, 6390, 7930, 9000])
