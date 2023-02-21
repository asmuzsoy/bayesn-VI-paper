# -*- coding: UTF-8 -*-
"""
BayeSN Plotting Utilities. Defines a set of functions which carry out some useful plotting operations.
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from astropy.table import unique

def kde(x_data, x_target, y_data=None, y_target=None, x_bounds=None, y_bounds=None, smoothing=1.0):
	"""
	Compute a one or two dimensional Kernel Density Estimate from MCMC samples.

	The internal code here is messy, and could do with tidying up. This is used as opposed to :py:class:`scipy.stats.gaussian_kde` so as to provide proper handling of hard boundaries via the reflection method [https://uk.mathworks.com/help/stats/ksdensity.html]. My implementation is not very fast.

	Parameters
	----------
	x_data : :py:class:`numpy.array`
		1D array containing MCMC samples of first parameter.
	x_target : :py:class:`numpy.array`
		1D array containing points along x axis at which KDE is to be evaluated.
	y_data : :py:class:`numpy.array`, optional
		1D array containing MCMC samples of second parameter (if required).
	y_target : :py:class:`numpy.array`, optional
		1D array containing points along y axis at which KDE is to be evaluated (if required).
	x_bounds : tuple, optional
		Upper and lower bounds in x direction. Pass None as one element if parameter is unbounded on that side.
	y_bounds : tuple, optional
		Upper and lower bounds in y direction.
	smoothing : float, optional
		Constant factor by which Scott's bandwidth rule is multiplied when making KDEs. Defaults to 1.
	
	Returns
	-------
	f : :py:class:`numpy.array`
		Kernel density estimate of PDF evaluated at x_target (,y_target)
	"""
	if y_data is None:
		n = len(x_data)
		d = 1
	else:
		if len(x_data) == len(y_data):
			n = len(x_data)
			d = 2
		else:
			raise ValueError("Data vectors should be same length.")
	b = smoothing*n**(-1./(d+4)) #Scott Rule x Smoothing Factor
	if d==1:
		h = np.std(x_data)*b
	else:
		h = np.cov([x_data, y_data])*b**2

	x = x_target[:,None] - x_data[None,:]
	if d==2:
		y = y_target[:,None] - y_data[None,:]
		KH = scipy.stats.multivariate_normal.pdf(np.stack([x,y], axis=-1), cov=h)
		if x_bounds is not None:
			if x_bounds[0] is not None:
				x_minus = 2*x_bounds[0] - x_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_minus[None,:], y], axis=-1), cov=h)
				if y_bounds is not None:
					if y_bounds[0] is not None:
						y_minus = 2*y_bounds[0] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_minus[None,:], y_target[:,None] - y_minus[None,:]], axis=-1), cov=h)
					if y_bounds[1] is not None:
						y_plus = 2*y_bounds[1] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_minus[None,:], y_target[:,None] - y_plus[None,:]], axis=-1), cov=h)
			if x_bounds[1] is not None:
				x_plus = 2*x_bounds[1] - x_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_plus[None,:], y], axis=-1), cov=h)
				if y_bounds is not None:
					if y_bounds[0] is not None:
						y_minus = 2*y_bounds[0] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_plus[None,:], y_target[:,None] - y_minus[None,:]], axis=-1), cov=h)
					if y_bounds[1] is not None:
						y_plus = 2*y_bounds[1] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_plus[None,:], y_target[:,None] - y_plus[None,:]], axis=-1), cov=h)
		if y_bounds is not None:
			if y_bounds[0] is not None:
				y_minus = 2*y_bounds[0] - y_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x, y_target[:,None] - y_minus[None,:]], axis=-1), cov=h)
			if y_bounds[1] is not None:
				y_plus = 2*y_bounds[1] - y_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x, y_target[:,None] - y_plus[None,:]], axis=-1), cov=h)
		f = np.sum(KH, axis=1)/n
	else:
		Kh = scipy.stats.norm.pdf(x, scale=h)
		if x_bounds is not None:
			if x_bounds[0] is not None:
				x_minus = 2*x_bounds[0] - x_data
				Kh += scipy.stats.norm.pdf(x_target[:,None] - x_minus[None,:], scale=h)
			if x_bounds[1] is not None:
				x_plus = 2*x_bounds[1] - x_data
				Kh += scipy.stats.norm.pdf(x_target[:,None] - x_plus[None,:], scale=h)
		f = np.sum(Kh, axis=1)/n

	return f

def corner(chains, names, colour="#1f77b4", lims=None, bounds=None, title=None, fig_ax=None, show_summary=True, figsize=None, smoothing=2.0, warn_tolerance=0.05):
	"""
	Produce a corner plot from a list of MCMC chains.

	The internal code here is kinda messy, and could do with tidying up. This will attempt to find the KDE contours containing 68 and 95% of the MCMC samples. If will print a WARNING if the fraction of samples in either of these is out by more than `warn_tolerance`.

	Parameters
	----------
	chains : list of :py:class:`numpy.array`
		List or array containing the MCMC samples. Each row should contain the samples for a different parameter.
	names : list of str
		List of parameter names. Should be equal in length to the first dimension of chains.
	colour : str, optional
		A valid matplotlib colour
	lims : None or array-like, optional
		An array or list of lists with two columns for each parameter, giving axis limits. If not provided, defaults to the range spanned by the MCMC samples (this may crop a bit early if the KDEs are very smoothed).
	bounds : None, array-like, or list of tuples, optional
		An array or list of lists/tuples giving hard axis bounds. These should be passed for parameters where a prior with a hard cutoff was imposed. In this case, the reflection method [https://uk.mathworks.com/help/stats/ksdensity.html] will be used to prevent mis-estimation of the KDE close to the boundary.
	title : str or None, optional
		An optional title for the whole figure
	fig_ax : tuple containing (:py:class:`matplotlib.figure.Figure`, array of :py:class:`matplotlib.axes.Axes`), optional
		A pair of Figure and Axes objects, as returned by :py:func:`matplotlib.pyplot.subplots` or this function. Allows you to plot onto an extant corner plot
	show_summary : bool, optional
		Turn on or off axis titles showing `"mean +/- std"` for each parameter.
	figsize : tuple or None, optional
		Figsize tuple. If not provided, defaults to a square figure with side of length 3 times the number of parameters.
	smoothing : float, optional
		Constant factor by which Scott's bandwidth rule is multiplied when making KDEs. Defaults to 2.
	warn_tolerance : float, optional
		Acceptable error in the fraction of MCMC samples in the 68 and 95% contours. A WARNING is issued outside this tolerance. Default is 0.05.
	
	Returns
	-------
	fig : :py:class:`matplotlib.figure.Figure`
		Figure object containing the plot
	ax : array of :py:class:`matplotlib.axes.Axes`
		Array containing the axes of the figure, tiled correctly. Exactly the same format as returned by :py:func:`matplotlib.pyplot.subplots`, and exactly the format which should be passed back into `corner` on a subsequent call.
	"""
	if len(chains) != len(names):
		raise ValueError("First dimension of input list/array should equal number of parameter names.")
	d = len(names)
	n = len(chains[0])
	if figsize is None:
		figsize = (3*d, 3*d)
	if lims is None:
		lims = [[None,None] for _ in range(d)]
	if bounds is None:
		bounds = [[None,None] for _ in range(d)]
	for p in range(d):
		if lims[p][0] is None:
			lims[p][0] = min(chains[p])
		if lims[p][1] is None:
			lims[p][1] = max(chains[p])


	if fig_ax is None:
		fig, ax = plt.subplots(d, d, figsize=figsize, sharex="col", sharey=False)
	else:
		fig = fig_ax[0]
		ax = fig_ax[1]
	for a in ax[np.triu_indices(d, k=1)]:
		a.axis("off")
	for row in range(d):
		pyrange = np.linspace(lims[row][0] - (bounds[row][0] is not None)*(lims[row][1]-lims[row][0]), lims[row][1] + (bounds[row][1] is not None)*(lims[row][1]-lims[row][0]), 100*int(1 + (bounds[row][0] is not None) + (bounds[row][1] is not None)))
		ax[row,row].plot(pyrange, kde(chains[row], pyrange, x_bounds=bounds[row], smoothing=smoothing), color=colour)
		if row == d-1:
			if lims is not None:
				ax[row,row].set_xlim(*lims[row])
		if show_summary:
			ax[row,row].set_title(names[row] + " = {:.3f} $\pm$ {:.3f}".format(np.mean(chains[row]), np.std(chains[row])), fontsize=11)
		ax[row,row].set_yticklabels("")
		ax[row,row].set_yticks([])
		for col in range(row):
			ax[row,col].get_shared_y_axes().remove(ax[row,row])
			pxrange = np.linspace(lims[col][0] - (bounds[col][0] is not None)*(lims[col][1]-lims[col][0]), lims[col][1] + (bounds[col][1] is not None)*(lims[col][1]-lims[col][0]), 100*int(1 + (bounds[col][0] is not None) + (bounds[col][1] is not None)))
			pxgrid, pygrid = np.meshgrid(pxrange, pyrange)
			cons = ax[row,col].contour(pxgrid, pygrid, np.reshape(kde(chains[col], pxgrid.flatten(), chains[row], pygrid.flatten(), x_bounds=bounds[col], y_bounds=bounds[row], smoothing=smoothing), pxgrid.shape), levels=25, colors=colour, alpha=0.1)
			fracs = []
			for c, con in enumerate(cons.collections):
				paths = con.get_paths()
				if len(paths) == 1:
					fracs.append(sum(paths[0].contains_points(np.vstack([chains[col], chains[row]]).T))/n)
				elif len(paths) == 0:
					fracs.append(np.inf)
				else:
					fracs.append(sum([sum(path.contains_points(np.vstack([chains[col], chains[row]]).T)) for path in paths])/n)
			c68 = np.fabs(np.array(fracs) - 0.68).argmin()
			c95 = np.fabs(np.array(fracs) - 0.95).argmin()
			if not 0.68 - warn_tolerance < fracs[c68] < 0.68 + warn_tolerance:
				print("WARNING: Fraction of samples contained in estimated ({}, {}) 68 percent credible interval is {:.3f}, plotted contour may be suspect!".format(names[col], names[row],fracs[c68]))
			if not 0.95 - warn_tolerance < fracs[c95] < 0.95 + warn_tolerance:
				print("WARNING: Fraction of samples contained in estimated ({}, {}) 95 percent credible interval is {:.3f}, plotted contour may be suspect!".format(names[col], names[row], fracs[c95]))
			ax[row,col].contour(pxgrid, pygrid, np.reshape(kde(chains[col], pxgrid.flatten(), chains[row], pygrid.flatten(), x_bounds=bounds[col], y_bounds=bounds[row], smoothing=smoothing), pxgrid.shape), levels=[cons.levels[c95], cons.levels[c68]], colors=colour)
			if col == 0:
				ax[row,col].set_ylabel(names[row])
				if lims is not None:
					ax[row,col].set_ylim(*lims[row])
			else:
				ax[row,col].set_yticklabels("")
				ax[row,col].set_yticks([])
				ax[row,col].set_ylim(ax[row,0].get_ylim())
			if row == d-1:
				ax[row,col].set_xlabel(names[col])
				if lims is not None:
					ax[row,col].set_xlim(*lims[col])
	ax[d-1,d-1].set_xlabel(names[d-1])
	if title is not None:
		fig.suptitle(title)
	fig.subplots_adjust(top=0.9)
	fig.subplots_adjust(wspace=0.075, hspace=0.075)

	return fig, ax


def plot_light_curve_fit(lc, samples, model, filters, colours=None, offsets=None, plot_snake=True, plot_draws=False, z=None, ebv_mw=None, logM=None, tmax=None, snr_cut=3, first_char_match=True, ax=None, figsize=(6.4, 4.8)):
	"""
	Plot a fitted light curve.

	This takes some MCMC chains and plots the light curve fit (in magnitudes) against some data. Can plot the fit as either an "error snake" (i.e. posterior mean light curve and its standard deviation) or a handful of posterior samples (or both).

	Less customiseable than doing it yourself.

	Parameters
	----------
	lc : :py:class:`astropy.table.Table`
		Astropy table containing light curve and (optionally) metadata. Same format/info as required by :py:func:`BayeSNmodel.bayesn_model.SEDmodel.fit_supernova`, etc.
	samples : dict
		Dictionary of MCMC samples, in the format returned by :py:func:`BayeSNmodel.bayesn_model.SEDmodel.fit_supernova`.
	model : :py:class:`BayeSNmodel.bayesn_model.SEDmodel`
		Model class instance which was used for the fit.
	filters : list
		List of passbands to plot the posteriors for. They will appear in this order in the legend, and will be plotted from the top down in this order unless custom offsets are provided. It is recommended that these are provided with the longest wavelength filter at the start of the list.
	colours : list, optional
		List of colours (in same order as filters). If not given, the function will assign something, which may give undesireable results.
	offsets : list, optional
		List of magnitude offsets to apply for clarity (in same order as filters). If not given, the function will attempt to guess something sensible, which may give undesireable results.
	plot_snake : bool, optional
		If True, plots an "error snake" w/ mean and standard deviation of posterior predictive light curves. Default is True.
	plot_draws : bool or int, optional
		If True, plots 10 draws from the light curve posterior distribution. If int, plots that many draws instead. If False, doesn't plot any. Default is False.
	z : float or None, optional
			Heliocentric redshift used for redshifting SED. Doesn't need to
			be specified explicitly if contained in `lc.meta` as ``'REDSHIFT_HELIO'``.
	ebv_mw : float or None, optional
		Milky Way reddening. Needn't be given if included in `lc.meta` as
		``'MWEBV'``.
	logM : float or None, optional
		log_10 of host galaxy stellar mass (only needed when using a model with
		a host galaxy mass split).
	tmax : float or None, optional
		Time of B-band maximum in MJD. If tmax was fixed in fits, this will be
		used as the time of maximum. If tmax was fit for, it will be assumed that
		this is the initial guess which was used. Needn't be given if included in `lc.meta` as ``'PEAKMJD'`` or	``'SEARCH_PEAKMJD'``.
	snr_cut : float, optional
		Minimum S/N cut applied to light curves before fitting. Default=3.
	first_char_match : bool, optional
		If True, for a filter in `filters`, will plot any data with a passband whose first initial matches the current filter's first initial. If False, will only plot strictly matching data. Default is True.
	ax : :py:class:`matplotlib.axes.Axes`, optional
		If given, will plot to this axes, instead of a new figure. Useful if constructing a figure with many subplots.
	figsize : tuple, optional
		Figure size, if creating a new figure. Default is (6.4, 4.8).
	
	Returns
	-------
	fig : :py:class:`matplotlib.figure.Figure`
		Figure object containing the plot
	ax : :py:class:`matplotlib.axes.Axes`
		Axis object containing the plot.
	"""

	#Check for heliocentric redshift
	if z is None:
		try:
			z = lc.meta["REDSHIFT_HELIO"]
		except KeyError as e:
			raise Exception("Redshift not provided or found in light curve metadata.") from e
	#Check for MW reddening
	if ebv_mw is None:
		try:
			ebv_mw = lc.meta["MWEBV"]
		except KeyError as e:
			raise Exception("Milky Way reddening not provided or found in light curve metadata.") from e
	#Check for hostmass (if necessary)
	if model.M_split is not False and logM is None:
		try:
			logM = lc.meta["HOSTGAL_LOGMASS"]
		except KeyError as e:
			raise Exception("Host galaxy mass not provided or found in light curve metadata") from e
	#Check for tmax
	if tmax is None:
		try:
			tmax = lc.meta["SEARCH_PEAKMJD"]
		except KeyError:
			try:
				tmax = lc.meta["PEAKMJD"]
			except KeyError as e:
				raise Exception("Time of maximum not provided or found in light curve metadata.") from e

	#See if tmax is free
	if "tmax" in samples.keys():
		tmax = np.mean(samples["tmax"])

	#Computes phase of observations for comparison, masks to [-10,40]d,
	#and applies S/N cut
	lc["phase"] = (lc["mjd"] - tmax)/(1 + z)
	mask = (lc["phase"] > min(model.t_k))*(lc["phase"] < max(model.t_k))*(lc["fluxcal"]/lc["fluxcalerr"] > snr_cut)
	lc = lc[mask]

	#Computes magnitudes and magnitude errors
	lc["mag"] = lc["zptmag"] - 2.5*np.log10(lc["fluxcal"])
	lc["magerr"] = np.fabs((2.5/np.log(10))*lc["fluxcalerr"]/lc["fluxcal"])

	#Sort out filter/colour/offset asignments
	if offsets is None:
		default_offsets = {"U": 4, "B": 2, "G": 0, "V": 0, "R": -2, "I": -4, "Z": -6, "Y": -5.5, "J": -8.5, "H": -10}
		offsets = []
		extra_off = 0
		if first_char_match is True:
			for flt in filters:
				if flt[0].upper() in default_offsets.keys():
					if len(offsets) > 0 and offsets[-1] > default_offsets[flt[0].upper()]:
						extra_off +=2
					offsets.append(default_offsets[flt[0].upper()] + extra_off)
				else:
					print("WARNING: One of the requested filters, {}, does not have a default colour/vertical offset - plotting outcome may be undesireable!".format(flt))
					extra_off += 2
					offsets.append(offsets[-1] + extra_off)
	if colours is None:
		default_colours = {"U": "blueviolet", "B": "b", "G": "springgreen", "V": "g", "R": "r", "I": "m", "Z": "darkviolet", "Y": "c", "J": "dodgerblue", "H": "navy"}
		colours = []
		for flt in filters:
			if flt[0].upper() in default_colours.keys():
				colours.append(default_colours[flt[0].upper()])
			else:
				colours.append(None)

	#Picks some indices along the MCMC chains
	if plot_draws is not False:
		if plot_draws is True:
			n_draws = 10
		else:
			if plot_draws <= len(samples["lp__"]):
				n_draws = plot_draws
			else:
				raise ValueError("Can only plot a number of draws less than or equal to the number of MCMC samples ({})".format(len(samples["lp__"])))
		inds = np.random.choice(len(samples["lp__"]), n_draws, replace=False)

	#Setup figure, if axis not provided
	if ax is None:
		fig = plt.figure(figsize=figsize)
		ax = plt.gca()
	else:
		fig = ax.get_figure()

	#Loops over filters
	for f, flt in enumerate(filters):
		#Plot an error snake, if requested
		if plot_snake is True:
			mags = []
			#Loops over MCMC samples
			for i in range(len(samples["lp__"])):
				#Simulates a light curve for this entry in the MCMC chain
				phase, mag = model.simulate_light_curve(flt, z=z, mu=samples["mu"][i], ebv_mw=ebv_mw, del_M=(samples["delM"])[i], AV=samples["AV"][i], theta=samples["theta"][i], epsilon=samples["epsilon"][i], mag=True)
				mags.append(mag)
			mags = np.array(mags)
			#Plots the mean light curve and error envelope
			ax.plot(phase, np.mean(mags, axis=0) + offsets[f], "k")
			ax.plot(phase, np.mean(mags, axis=0) + np.std(mags, axis=0) + offsets[f], "k--")
			ax.plot(phase, np.mean(mags, axis=0) - np.std(mags, axis=0) + offsets[f], "k--")
		#Plot draws, if requested
		if plot_draws is True:
			#Loops over MCMC samples
			for i in inds:
				#Simulates a light curve for this entry in the MCMC chain
				phase, mag = model.simulate_light_curve(flt, z=z, mu=samples["mu"][i], ebv_mw=ebv_mw, del_M=(samples["delM"])[i], AV=samples["AV"][i], theta=samples["theta"][i], epsilon=samples["epsilon"][i], mag=True)
				#Plots the simulated light curve
				ax.plot(phase, mag+offsets[f], color="gray", alpha=0.5)
		#Mask which finds similar filters in this light curve
		if first_char_match is True:
			flt_mask = lc["flt"].astype('<U1') == flt[0]
		else:
			flt_mask = lc["flt"] == flt
		#Plot data
		ax.errorbar(lc["phase"][flt_mask], lc["mag"][flt_mask]+offsets[f], yerr=lc["magerr"][flt_mask], marker=".", markersize=10, linestyle="", label=flt[0], color=colours[f])
	#Labels etc.
	ax.invert_yaxis()
	ax.set_xlabel("phase")
	ax.set_ylabel("mag + const.")
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	
	return fig, ax
