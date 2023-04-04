from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')
model.process_dataset('YSE_DR1', 'data/lcs/tables/YSE_DR1_table.txt', 'data/lcs/meta/YSE_DR1_meta.txt', data_mode='mag')
model.analyse_fit_sample('YSE_fit')
# model.plot_fits('YSE_fit')
