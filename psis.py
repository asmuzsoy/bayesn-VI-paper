import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import arviz.stats.stats as ass
# Author: Stephen Thorp


# this is a slightly contrived example
# target distribution
p = ss.t(loc=0, scale=1, df=7)
# proposal/approximate distribution
q = ss.norm(loc=0.2, scale=0.9)

# plot the two distributions
x = np.linspace(-3, 3, 1000)
plt.plot(x, p.pdf(x), label="$p(x)$")
plt.plot(x, q.pdf(x), label="$q(x)$")
plt.xlabel("$x$")
plt.ylabel("probability density")
plt.legend()
plt.show()

# take a load of samples from the proposal
S = 10000
x = q.rvs(size=S)
# compute importance weights
p_x = p.pdf(x)
q_x = q.pdf(x)
r_x = p_x/q_x
# set M
M = -int(np.ceil(np.min([S/5.0, 3.0*np.sqrt(S)]))) - 1

# do PSIS using a secret arviz function
lw, k = ass._psislw(np.log(r_x), M, np.log(np.finfo(float).tiny))
w = np.exp(lw)

# compute some expectations using the importance sampling estimator
# (eq. 3 in Yao+18)
mean_psis = np.sum(x*w)/np.sum(w)
var_psis = np.sum(w*(x - mean_psis)**2)/np.sum(w)
# compute true mean and variance
mean_true = p.mean()
var_true = p.var()
mean_prop = q.mean()
var_prop = q.var()
# naive importance sampling estimates
mean_is = np.sum(x*r_x)/np.sum(r_x)
var_is = np.sum(r_x*(x - mean_is)**2)/np.sum(r_x)

# print the results
print("Results\n-------")
print("k = {:.3f}".format(k))
print("true mean = {:.3f}\npsis mean = {:.3f}\nprop mean = {:.3f}\nnaiv mean = {:.3f}".format(mean_true, mean_psis, mean_prop, mean_is))
print("true variance = {:.3f}\npsis variance = {:.3f}\nprop variance = {:.3f}\nnaiv variance = {:.3f}".format(var_true, var_psis, var_prop, var_is))