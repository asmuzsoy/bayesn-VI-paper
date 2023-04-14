import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
from bayesn_model import SEDmodel

model = SEDmodel(load_model='T21_model')

with open(f'model_files/T21_model/BAYESN.YAML', 'r') as file:
    params = yaml.load(file, Loader=yaml.Loader)
l_knots = np.array(params['L_KNOTS'])
tau_knots = np.array(params['TAU_KNOTS'])
W0 = np.array(params['W0'])
W1 = np.array(params['W1'])
L_Sigma = np.array(params['L_SIGMA_EPSILON'])
M0 = np.array(params['M0'])
sigma0 = np.array(params['SIGMA0'])
Rv = np.array(params['RV'])
tauA = np.array(params['TAUA'])

del_eps = 1e-6
n_eps = L_Sigma.shape[0]
all_eps = np.zeros((n_eps + 1, l_knots.shape[0], tau_knots.shape[0]))
for i in range(n_eps):
    eps = np.zeros(n_eps)
    eps[i] = del_eps
    eps = eps.reshape((l_knots.shape[0] - 2, tau_knots.shape[0]), order='F')
    full_eps = np.zeros((l_knots.shape[0], tau_knots.shape[0]))
    full_eps[1:-1, :] = eps
    all_eps[i + 1, ...] = full_eps

lc = model.simulate_light_curve([np.array([10])], n_eps + 1, ['g_PS1', 'r_PS1'], write_to_files=False,
                                theta=0, del_M=0, AV=0, eps=all_eps, mag=True)[0]
q = ((lc[0, 1:] - lc[1, 1:]) - (lc[0, 0] - lc[1, 0])) / del_eps
q_matrix = np.diag(q)
q = q[:, None]

# Calculate epsilon prime
sigma_eps = L_Sigma @ L_Sigma.T
sigma_err = 0.0001
sigma_c_2 = (q.T @ sigma_eps @ q + sigma_err * sigma_err)[0, 0]
print(np.sqrt(sigma_c_2))
Wc = sigma_eps @ q / sigma_c_2
Wc = np.reshape(Wc, (l_knots.shape[0] - 2, tau_knots.shape[0]), order='F')
plt.imshow(Wc)
plt.colorbar()
plt.show()
sigma_eps_prime = sigma_eps - sigma_eps @ q @ q.T @ sigma_eps / sigma_c_2

test = np.random.multivariate_normal(np.zeros(n_eps), sigma_eps_prime, size=100)
test = np.reshape(test, (100, l_knots.shape[0] - 2, tau_knots.shape[0]), order='F')
full_test = np.zeros((100, l_knots.shape[0], tau_knots.shape[0]))
full_test[:, 1:-1, :] = test

lc = model.simulate_light_curve([np.array([10])], 100, ['g_PS1', 'r_PS1'], write_to_files=False,
                                theta=0, del_M=0, AV=0, eps=full_test, mag=True)[0]
print(np.std(lc[0, :] - lc[1, :]))

print(np.sum(np.diagonal(sigma_eps)))
print(np.sum(np.diagonal(sigma_eps_prime)))
eig = np.linalg.eig(sigma_eps)[0]
eig_prime = np.linalg.eig(sigma_eps_prime)[0]
print(np.sum(eig))
print(np.sum(eig_prime))
plt.hist(eig, histtype='step', label='Epsilon')
plt.hist(eig_prime, histtype='step', label='Epsilon_prime')
plt.legend()
plt.show()

# Plot sigmaepsilon
fig, ax = plt.subplots(1, 3, figsize=(24, 8))
im = ax[0].imshow(sigma_eps)
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
ax[0].set_title(r"$\epsilon$")
im = ax[1].imshow(sigma_eps_prime)
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].set_title(r"$\epsilon '$")
im = ax[2].imshow(sigma_eps - sigma_eps_prime)
plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].set_title(r"$\epsilon - \epsilon '$")
plt.show()
