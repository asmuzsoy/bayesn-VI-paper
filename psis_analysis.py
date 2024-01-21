import numpy as np
from bayesn_model import SEDmodel
import numpy as np
from zltn_utils import *
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoLaplaceApproximation
import pandas as pd


best_ks = np.loadtxt("foundation_results/best_ks.txt")
last_ks = np.loadtxt("foundation_results/last_ks.txt")

plt.plot(best_ks, last_ks, 'o')
plt.plot(np.linspace(0.4, 1.2),np.linspace(0.5, 1.2), 'k')
plt.xlabel("k from best loss params")
plt.ylabel("k from last iteration params")
plt.show()


laplace_best_params = np.load("foundation_results/foundation_vmap_laplace_011724_bestparams.npy", allow_pickle=True).item()
laplace_best_samples = np.load("foundation_results/foundation_vmap_laplace_011724_samples.npy", allow_pickle=True).item()
zltn_best_params = np.load("foundation_results/foundation_vmap_zltn_011724_bestparams.npy", allow_pickle=True).item()
zltn_best_samples = np.load("foundation_results/foundation_vmap_zltn_011724_samples.npy", allow_pickle=True).item()
multinormal_best_params = np.load("foundation_results/foundation_vmap_multinormal_011724_bestparams.npy", allow_pickle=True).item()
multinormal_best_samples = np.load("foundation_results/foundation_vmap_multinormal_011724_samples.npy", allow_pickle=True).item()

model = SEDmodel(load_model='T21_model')

dataset = 'T21_training_set'


original_order = np.array(['AV', 'Ds', 'eps_tform', 'theta', 'tmax'])
desired_order = np.array(['AV', 'theta', 'tmax', 'eps_tform', 'Ds'])


lengths = {'AV':1, 'Ds':1, 'eps_tform':24, 'theta':1, 'tmax':1}

def reorder_params(params_dict):
	loc = params_dict['auto_loc']
	scale_tril = params_dict['auto_scale_tril']

	# new_loc = []
	# for i, param in enumerate(desired_order):
	# 	param_length = lengths[param]
	# 	current_location = np.where(original_order == param)[0][0]
	# 	for k in range(param_length):
	# 		new_loc.append(loc[current_location + k])
	# new_loc = np.squeeze(new_loc)

	num_params = len(loc)

	cov_matrix = np.matmul(scale_tril, scale_tril.T)
	new_cov_matrix = np.zeros_like(scale_tril)
	old_indexes = np.arange(num_params)
	new_indexes = np.hstack(([0], [27], np.arange(3,27), [1], [2]))
	print(old_indexes)
	print(new_indexes)
	index_dict = dict(zip(new_indexes,old_indexes)) # dict[new_index] = old_index

	new_loc = np.zeros(num_params)
	for i in range(num_params):
		new_loc[i] = loc[index_dict[i]]
	# 	for j in range(num_params):
	# 		# if scale_tril[index_dict[i]][index_dict[j]] == 0:
	# 		# 	new_scale_tril[i][j] = scale_tril[index_dict[j]][index_dict[i]]
	# 		# else:
	# 		new_cov_matrix[i][j] = cov_matrix[index_dict[i]][index_dict[j]]
	# 		print(cov_matrix[index_dict[i]][index_dict[j]], cov_matrix[index_dict[j]][index_dict[i]])
	# new_scale_tril = np.linalg.cholesky(new_cov_matrix)

	print(new_loc)
	# set up a permutation matrix
	P = np.zeros((num_params, num_params), dtype=int)
	P[new_indexes, old_indexes] = 1
	# permute the matrix
	new_cov_matrix = P.T @ cov_matrix @ P
	new_scale_tril = np.linalg.cholesky(new_cov_matrix + np.eye(num_params)*1e-3)
	# new_scale_tril = P.T @ scale_tril

	fig, ax = plt.subplots(1,2)
	ax[0].imshow(scale_tril)
	ax[1].imshow(new_scale_tril)
	plt.show()

	print(scale_tril[1][1], new_scale_tril[-1][-1])
	print(np.diagonal(cov_matrix))
	print(np.diagonal(new_cov_matrix))




	# cov_matrix = np.matmul(scale_tril, scale_tril.T)
	# print(scale_tril)
	# num_params = len(loc)
	# temp_scale_tril = np.copy(scale_tril)
	# for i in range(num_params):
	# 	for j in range(num_params):
	# 		if temp_scale_tril[i][j] == 0:
	# 			temp_scale_tril[i][j] = temp_scale_tril[j][i]

	# fig, ax = plt.subplots(2,2)
	# ax[0][0].imshow(scale_tril)

	# ax[0][1].imshow(temp_scale_tril)

	# new_scale_tril = np.zeros_like(scale_tril)
	# current_length1 = 0
	# for i, param1 in enumerate(desired_order):
	# 	print(param1)
	# 	param_length1 = lengths[param1]
	# 	original_location1 = np.where(original_order == param1)[0][0]
	# 	for l in range(param_length1):
	# 		current_length2 = 0
	# 		for j, param2 in enumerate(desired_order):
	# 			print(param2)
	# 			param_length2 = lengths[param2]
	# 			original_location2 = np.where(original_order == param2)[0][0]
	# 			for k in range(param_length2):
	# 				new_scale_tril[current_length1][current_length2] = temp_scale_tril[original_location1 + l][original_location2 + k]
	# 				current_length2 += 1
	# 				print(current_length1, current_length2)
	# 		current_length1 += 1

	# print("new:")
	# ax[1][0].imshow(new_scale_tril)

	# for i in range(num_params):
	# 	for j in range(i):
	# 		new_scale_tril[num_params-i -1][num_params-j - 1] = 0
	# ax[1][1].imshow(new_scale_tril)
	# plt.show()
	# print(x)
	return {'auto_loc':new_loc, 'auto_scale_tril':new_scale_tril}





sn_list = pd.read_csv('data/lcs/tables/' + dataset + '.txt', comment='#', delim_whitespace=True, names=['sn', 'source', 'files'])
sn_names = list(sn_list.sn.values)

sn_names[sn_names.index("AT2016ajl")] = 'AT2016aj'


filt_map_dict = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}


laplace_guide = AutoLaplaceApproximation(model.fit_model_mcmc)
zltn_guide = AutoMultiZLTNGuide(model.fit_model_vi)
multinormal_guide = AutoMultivariateNormal(model.fit_model_mcmc)


def get_ks(vi_model, guide, params, samples):
	samples['Ds'] = samples['Ds'][...,None]
	ks = []
	for i in range(157):
		print(sn_names[i])
		np.savetxt("temp_sn_list.txt", [sn_names[i]], fmt="%s", header="SNID", comments="")
		model.process_dataset('foundation', 'data/lcs/tables/' + dataset + '.txt', 'data/lcs/meta/' + dataset + '_meta.txt',
	                      filt_map_dict, data_mode='flux', sn_list = 'temp_sn_list.txt')

		this_sn_params = {k:params[k][i] for k in params.keys()}
		
		new_params = reorder_params(this_sn_params)
		this_sn_samples = {k:samples[k][i] for k in samples.keys()}
		k= model.psis(vi_model, guide, new_params, this_sn_samples)
		print("k", k)
		print(x)
		ks.append(k)
	return ks

## need to change to Gaussian for mcmc model too
ks = get_ks(model.fit_model_vi, zltn_guide, zltn_best_params, zltn_best_samples)
print(ks)