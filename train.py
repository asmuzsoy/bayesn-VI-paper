from model import Model

model = Model()
filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
#model.process_dataset('foundation', 'data/LCs/foundation/Foundation_DR1', 'data/LCs/meta/T21_training_set_meta.txt',
#                      filt_map_dict, sn_list='data/LCs/foundation/Foundation_DR1/Foundation_DR1.LIST')
model.process_dataset('ztf', 'data/LCs/ZTF', 'data/LCs/meta/ztf_dr1_training.txt',
                      map_dict=None)
model.train(1000, 1000, 4, 'ztf_train_test', chain_method='sequential', init_strategy='map',
            l_knots=[4150, 4760, 6390, 7930, 9000])
