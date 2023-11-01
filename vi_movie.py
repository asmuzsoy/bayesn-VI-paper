import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
from JSAnimation import IPython_display
import pickle
import numpy as np
import corner
from astropy.cosmology import FlatLambdaCDM
import imageio



fiducial_cosmology={"H0": 73.24, "Om0": 0.28}
cosmo = FlatLambdaCDM(**fiducial_cosmology)

known_values = {'AV': 0.1, 'mu':34.59932899, 'theta':2.}

lower_known_values = {'AV': 0.01, 'mu':34.59932899, 'theta':2.}

nonzero_eps_values = {'AV': 0.05, 'mu':34.59932899, 'theta':2.}


range1 = [(0,0.2), (34, 35), (1.5,2.5)]

for j in range(50):

	dataset = 'sim_low_AV_vi_' + str(j)
	# dataset = 'sim_zero_AV'


	with (open("results/" + dataset + "/chains.pkl", "rb")) as openfile:
		vi_objects = pickle.load(openfile)

	for i in range(1):
		mcmc_results = []
		vi_results = []
		for var in ['AV', 'mu', 'theta']:
			vi_samples = np.squeeze(vi_objects[var][:,i])
			print(vi_samples)
			vi_results.append(vi_samples)

		vi_results = np.array(vi_results).T

		fig = corner.corner(vi_results, labels = ["$A_V$", "$\\mu$", "$\\theta$"])
		
		if 'sim_low_AV' in dataset:
			corner.overplot_lines(fig, [known_values['AV'],known_values['mu'],known_values['theta']], linestyle = 'dashed', color='b')
		
		colors = ['k', 'b', 'w']

		labels = ['VI Samples', 'True Values', " \n Iterations: " + str(1000*(j+1))]

		plt.legend(
		    handles=[
		        mlines.Line2D([], [], color=colors[i], label=labels[i])
		        for i in range(len(labels))
		    ],
		    fontsize=14, frameon=False, bbox_to_anchor=(0.8, 3), loc="upper right"
		)

		plt.savefig("movie/" + str(j)+".png")
		# plt.show()


with imageio.get_writer('movie/movie.gif', mode='I', fps=2) as writer:
    for j in range(50):
        image = imageio.imread("movie/" + str(j)+".png")
        writer.append_data(image)