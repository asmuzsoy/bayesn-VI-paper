from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')

filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
model.process_dataset('foundation', 'data/lcs/tables/T21_training_set.txt', 'data/lcs/meta/T21_training_set_meta.txt',
                      filt_map_dict, data_mode='mag')

#model.process_dataset('T21_sim_1000', 'data/lcs/tables/T21_sim_1000.txt', 'data/lcs/meta/T21_sim_1000_meta.txt',
#                      data_mode='mag')

#model.process_dataset('M20', 'data/lcs/tables/M20_training_set.txt', 'data/lcs/meta/M20_training_set_meta.txt',
#                      data_mode='mag')

#model.process_dataset('ztf', 'data/lcs/ZTF', 'data/lcs/meta/ztf_dr1_training.txt',#            l_knots=[4150, 4760, 6390, 7930, 9000])
#                      map_dict=None, data_mode='mag')

#model.process_dataset('YSE_DR1', 'data/lcs/tables/YSE_DR1_table.txt', 'data/lcs/meta/YSE_DR1_meta.txt', data_mode='mag')

model.train(1000, 1000, 4, 'T21_popRv', chain_method='parallel', init_strategy='T21', max_tree_depth=10)

