from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')

dataset = 'sim_low_AV'
# dataset = 'sim_nonzero_eps'

# dataset = 'T21_sim_2'

filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
                      filt_map_dict, data_mode='flux')

#model.process_dataset('T21_sim_1000', 'data/lcs/tables/T21_sim_1000.txt', 'data/lcs/meta/T21_sim_1000_meta.txt',
#                      data_mode='flux')

#model.process_dataset('YSE_DR1', 'data/lcs/tables/YSE_DR1_table.txt', 'data/lcs/meta/YSE_DR1_meta.txt', data_mode='flux')

#model.process_dataset('M20', 'data/lcs/tables/M20_training_set.txt', 'data/lcs/meta/M20_training_set_meta.txt',
#                      data_mode='flux')

#filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
#model.process_dataset('YSE_full', 'data/lcs/tables/YSEfull_table.txt', 'data/lcs/meta/YSEfull_meta.txt', data_mode='flux',
#                      map_dict=filt_map_dict)

print("Fitting MCMC...")
model.fit(250, 250, 4, dataset + '_mcmc', chain_method='parallel', init_strategy='median')

print("Fitting VI...")
model.fit_with_vi(dataset + '_vi', init_strategy='median')
