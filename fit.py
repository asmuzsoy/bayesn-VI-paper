from bayesn_model import SEDmodel

model = SEDmodel()

#filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
#model.process_dataset('foundation', 'data/lcs/tables/T21_training_set.txt', 'data/lcs/meta/T21_training_set_meta.txt',
#                      filt_map_dict, data_mode='mag')

#model.process_dataset('T21_sim_10000', 'data/lcs/tables/T21_sim_10000.txt', 'data/lcs/meta/T21_sim_10000_meta.txt',
#                      data_mode='flux')

model.process_dataset('YSE_DR1', 'data/lcs/tables/YSE_DR1_table.txt', 'data/lcs/meta/YSE_DR1_meta.txt', data_mode='flux')

model.fit(250, 250, 4, 'YSE_fit_test', chain_method='parallel', init_strategy='median')
