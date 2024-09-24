import numpy as np
import pandas as pd
from sksurv.nonparametric import nelson_aalen_estimator
from scipy.optimize import minimize
from src.prediction import predict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')

def individual_conditional_expectation(explainer, selected_features, n_sel_samples = 100,
                                       n_grid_points = 50, type = "survival"):
	"""
    Compute individual conditional expectation (ICE)


    Parameters
    ----------
    explainer : `class`
		A Python class used to explain the survival model

    selected_features :  `str`
        Name of the desired features to be explained

    n_sel_samples :  `int`, default = 100
		Number of observations used for the caculation of aggregated profiles

    n_grid_points :  `int`, default = 50
		Number of grid points used for the caculation of aggregated profiles

    type :  `str`, default = "survival"
		The character of output type, either "risk", "survival" or "chf" depending
        on the desired output


    Returns
    -------
    res : `np.ndarray`, shape=(n_sel_samples, )
        The ICE values of selected features for desired observations
	"""
	data = explainer.data
	sel_feat_idx = data.columns.get_loc(selected_features)
	n_samples = data.shape[0]
	n_sel_samples = min(n_sel_samples, n_samples)
	X_sel = data[:n_sel_samples].values
	X_feat_sel = X_sel[:, sel_feat_idx]
	# Support uniform gird of points
	X_feat_sel_min, X_feat_sel_max = min(X_feat_sel), max(X_feat_sel)
	X_ext = np.repeat(X_sel, n_grid_points, axis=0)
	X_space = np.linspace(X_feat_sel_min, X_feat_sel_max, n_grid_points)
	X_replace = np.repeat(X_space, n_sel_samples).reshape(
		(n_grid_points, n_sel_samples)).T.flatten()
	X_ext[:, sel_feat_idx] = X_replace

	data_ext = pd.DataFrame(data=X_ext, columns=data.columns.values)

	if type == "survival":
		pred = predict(explainer, data_ext)
	elif type == "chf":
		pred = predict(explainer, data_ext, "chf")
	else:
		raise ValueError("Unsupported")

	ICE_df = pd.DataFrame(columns=["id", "X", "times", "pred"])
	for i in range(n_sel_samples):
		for j in range(n_grid_points):
			pred_ij = pred[pred.id == float(i * n_grid_points + j)]
			n_pred_times = pred_ij.shape[0]
			for k in range(n_pred_times):
				ICE_df.loc[len(ICE_df)] = [i, X_space[j], pred_ij.times.values[k], pred_ij.pred.values[k]]

	return ICE_df

def plot_ICE(res, explained_feature = "", id=0):
	"""
	Visualize the ICE results

	Parameters
    ----------
    res : `pd.Dataframe`
		ICE result to be visualize
	"""
	pred_times = np.unique(res.times.values)
	n_pred_times = len(pred_times)

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	times_norm = (pred_times - min(pred_times)) / (max(pred_times) - min(pred_times))
	cmap = mpl.cm.ScalarMappable(
		norm=mpl.colors.Normalize(0.0, max(pred_times), True), cmap='BrBG')
	for i in np.arange(0, n_pred_times):
		res_i = res.loc[(res.id == id) & (res.times == pred_times[i])]
		sns.lineplot(data=res_i, x="X", y="pred",
		             color=cmap.get_cmap()(times_norm[i]))

	ax.set_ylim(0, 1)
	plt.xlabel("")
	plt.ylabel("Survival prediction")
	plt.title("ICE for feature {0} of obsevation id = {1}".format(explained_feature, id))
	plt.colorbar(cmap, orientation='horizontal', label='Time')
	plt.show()


def counterfactual_explanations(explainer):
	"""
	Compute Counterfactual Explanations


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	raise ValueError("Not supported yet")

def random_ball(n_points, n_dims, radius=1.):
	"""
	Create a random points supported for SurvLIME

    Parameters
    ----------
    n_points :  `int`
		Number of points to be generated

    n_dims :  `int`
		Number of dimensions (features)

    radius :  `float`, default = 1.
		The radius

    Returns
    -------
    res : `np.ndarray`, shape=(num_points, n_dims)
        The generated random points
	"""

	# First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
	rand_dir = np.random.normal(size=(n_dims, n_points))
	rand_dir /= np.linalg.norm(rand_dir, axis=0)
	# Second generate a random radius with probability proportional to
	# the surface area of a ball with a given radius.
	rand_radius = np.random.random(n_points) ** (1 / n_dims)

	# Return the list of random (direction & length) points.
	res = radius * (rand_dir * rand_radius).T

	return res

def SurvLIME(explainer, data, label, n_nearest=100, id=None):
	"""
	Compute SurvLIME


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model

	data :  `np.ndarray`, shape=(n_samples, n_features)
		New observations for which predictions need to be explained

	label : `np.ndarray`, shape=(n_samples, 2)
		Survival labels of new observations

	n_nearest : `int`, default = 100
		Number of neighbor points of the interest observations to be generated

	Returns
	-------
	SurvLIME_df : `pd.Dataframe`
		The SurvLIME results
	"""

	def objective(x, model_chf, baseline_chf, neibourg_points, distance_weights, straightening_weights, unique_times):

		dt = unique_times[1:] - unique_times[:-1]
		tmp = np.sum((straightening_weights[:, :-1] ** 2) * ((np.log(model_chf[:, :-1]) - np.log(baseline_chf[:-1]) - neibourg_points.dot(x).reshape((-1, 1))) ** 2 * dt), axis=1)
		f_x = np.sum(distance_weights * tmp)

		return f_x

	surv_time = label[:, 0]
	surv_ind = label[:, 1].astype('bool')
	# cumulative hazard function
	unique_times, baseline_chf = nelson_aalen_estimator(surv_ind, surv_time)

	# TODO: Update for all individuals
	if id is None:
		id = 0
	if isinstance(data, pd.DataFrame):
		interest_point = data.iloc[id].values
	else:
		interest_point = data[id]
	n_feats = data.shape[1]
	radius = 1
	neibourg_points = interest_point + random_ball(n_nearest, n_feats, radius)

	model_chf = predict(explainer, neibourg_points, unique_times, type="chf")
	distance_weights = 1 - np.sqrt(np.linalg.norm(neibourg_points - interest_point, axis=1) / radius)
	straightening_weights = model_chf / np.log(model_chf)

	init_point = np.random.normal(size = n_feats)
	args = (model_chf, baseline_chf, neibourg_points, distance_weights, straightening_weights, unique_times)
	coefs = minimize(objective, init_point, args=args, method='BFGS')["x"]

	SurvLIME_res = np.stack([data.columns.values, coefs]).T
	SurvLIME_df = pd.DataFrame(data = SurvLIME_res, columns=["feats", "coefs"]).reset_index(drop=True)

	return SurvLIME_df

def plot_SurvLIME(res, id=0):
	"""
	Visualize the ICE results

	Parameters
	----------
	res : `pd.Dataframe`
		SurvLIME result to be visualize
	"""

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	colors = ['g' if c >= 0 else 'r' for c in res.coefs.values]
	sns.barplot(data=res, y="feats", x="coefs", palette=colors)

	plt.xlabel("Coefs")
	plt.ylabel("")
	plt.title("SurvLIME of obsevation id = {}".format(id))
	plt.show()

def SurvSHAP(explainer):
	"""
	Compute SurvSHAP


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	raise ValueError("Not supported yet")