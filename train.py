from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')

#filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
#model.process_dataset('foundation', 'data/lcs/tables/T21_training_set.txt', 'data/LCs/meta/T21_training_set_meta.txt',
#                      filt_map_dict, data_mode='mag')

#model.process_dataset('T21_sim_10000', 'data/lcs/tables/T21_sim_10000.txt', 'data/LCs/meta/T21_sim_10000_meta.txt',
#                      data_mode='mag')

#model.process_dataset('M20', 'data/lcs/tables/M20_training_set.txt', 'data/lcs/meta/M20_training_set_meta.txt',
#                      data_mode='mag')

#model.process_dataset('ztf', 'data/LCs/ZTF', 'data/LCs/meta/ztf_dr1_training.txt',#            l_knots=[4150, 4760, 6390, 7930, 9000])
#                      map_dict=None, data_mode='mag')

#model.train(500, 500, 4, 'T21_sim_10000_train', chain_method='sequential', init_strategy='median')

