import numpy as np
import pandas as pd
from sksurv.nonparametric import nelson_aalen_estimator
from scipy.optimize import minimize
from src.prediction import predict
import seaborn as sns
import sklearn
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
	if selected_features in explainer.numeric_feats:
		sel_feat_idx = data.columns.get_loc(selected_features)
		n_samples = data.shape[0]
		n_sel_samples = min(n_sel_samples, n_samples)
		X_sel = data[:n_sel_samples].values
		X_feat_sel = X_sel[:, sel_feat_idx]
		# Support uniform gird of points
		X_feat_sel_min, X_feat_sel_max = min(X_feat_sel), max(X_feat_sel)
		X_ext = np.repeat(X_sel, n_grid_points, axis=0)
		X_space = np.linspace(X_feat_sel_min, X_feat_sel_max, n_grid_points)
		X_replace = np.repeat(X_space, n_sel_samples).reshape((n_grid_points, n_sel_samples)).T.flatten()
		X_ext[:, sel_feat_idx] = X_replace

		data_ext = pd.DataFrame(data=X_ext, columns=data.columns.values)

	else:
		cate_features_ext = [feat for feat in data.columns.values if selected_features in feat]
		sel_feat_idx = [data.columns.get_loc(feat) for feat in cate_features_ext]
		n_samples = data.shape[0]
		n_sel_samples = min(n_sel_samples, n_samples)
		X_space = data[cate_features_ext].drop_duplicates().values
		n_unique = X_space.shape[0]
		X_sel = data[:n_sel_samples].values
		X_ext = np.repeat(X_sel, n_unique, axis=0)
		X_replace = np.tile(X_space, (n_sel_samples, 1))
		X_ext[:, sel_feat_idx] = X_replace
		data_ext = pd.DataFrame(data=X_ext, columns=data.columns.values)

	if type == "survival":
		pred = predict(explainer, data_ext)
	elif type == "chf":
		pred = predict(explainer, data_ext, "chf")
	else:
		raise ValueError("Unsupported")

	if selected_features in explainer.numeric_feats:
		ICE_df = pd.DataFrame(columns=["id", "times", "pred", selected_features])
	else:
		ICE_df = pd.DataFrame(columns=["id", "times", "pred"] + cate_features_ext)

	for i in range(n_sel_samples):
		if selected_features in explainer.numeric_feats:
			for j in range(n_grid_points):
				pred_ij = pred[pred.id == float(i * n_grid_points + j)]
				n_pred_times = pred_ij.shape[0]
				for k in range(n_pred_times):
						ICE_df.loc[len(ICE_df)] = [i, pred_ij.times.values[k], pred_ij.pred.values[k]] + [X_space[j]]
		else:
			for j in range(n_unique):
				pred_ij = pred[pred.id == float(i * n_unique + j)]
				n_pred_times = pred_ij.shape[0]
				for k in range(n_pred_times):
					ICE_df.loc[len(ICE_df)] = [i, pred_ij.times.values[k], pred_ij.pred.values[k]] + X_space[j].tolist()

	if explainer.cate_feats is not None:
		if selected_features in explainer.cate_feats:
			encoder = explainer.encoders[selected_features]
			feat_col_sel = encoder.get_feature_names_out([selected_features]).tolist()
			ICE_df[selected_features] = encoder.inverse_transform(ICE_df[feat_col_sel].values).flatten()

	return ICE_df

def plot_ICE(explainer, res, explained_feature = "", id=0):
	"""
	Visualize the ICE results

	Parameters
    ----------
    res : `pd.Dataframe`
		ICE result to be visualize
	"""

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	if explained_feature in explainer.numeric_feats:
		X_unique = np.unique(res[explained_feature].values)
		n_unique = len(X_unique)
		X_norm = (X_unique - min(X_unique)) / (max(X_unique) - min(X_unique))
		cmap = mpl.cm.ScalarMappable(
			norm=mpl.colors.Normalize(0.0, max(X_unique), True), cmap='BrBG')
		for i in np.arange(0, n_unique):
			res_i = res.loc[(res.id == id) & (res[explained_feature] == X_unique[i])]
			sns.lineplot(data=res_i, x="times", y="pred", color=cmap.get_cmap()(X_norm[i]))

		plt.colorbar(cmap, orientation='vertical', label=explained_feature, ax=ax)
	else:
		sel_res = res[res.id == id].sort_values(by=explained_feature)
		sns.lineplot(data=sel_res, x="times", y="pred", hue=explained_feature)

	ax.set_ylim(0, 1)
	plt.xlabel("Time")
	plt.ylabel("Survival prediction")
	plt.title("ICE for feature {0} of obsevation id = {1}".format(explained_feature, id))
	plt.savefig("ICE_feature_{0}_of_id={1}.pdf".format(explained_feature, id), bbox_inches='tight')
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


def neighbor_generation(explainer, data, n_points):
	"""
	Create a random points supported for SurvLIME

	Parameters
	----------
	n_points :  `int`
	    Number of points to be generated

	Returns
	-------
	res : `np.ndarray`, shape=(num_points, n_dims)
	    The generated random points
	"""

	np.random.seed(seed=0)
	n_feats = data.shape[1]
	neighbor = np.random.normal(size=(n_points, n_feats))
	neighbor_df = pd.DataFrame(data=neighbor, columns=data.columns.values)
	scale_neighbor_df = neighbor_df.copy(deep=True)
	data_features = explainer.numeric_feats + explainer.cate_feats
	training_data = explainer.data
	scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
	scaler.fit(training_data)
	scale = scaler.scale_
	mean = scaler.mean_
	idx_ext = 0
	for idx in range(len(data_features)):
		feature = data_features[idx]
		if feature in explainer.numeric_feats:
			sample = neighbor_df[feature].values * scale[idx] + mean[idx]
			neighbor_df[feature] = sample
			scale_neighbor_df[feature] = sample
			idx_ext += 1
		else:
			cate_features_ext = [feat for feat in data.columns.values if feature in feat]
			training_data_counts = training_data.groupby(cate_features_ext).value_counts().reset_index(name='counts')
			cate_neighbor = training_data_counts.sample(n_points, replace=True, weights="counts", random_state=1)[cate_features_ext].values
			#cate_neighbor = training_data_counts.sample(n_points, replace=True, weights="counts")[cate_features_ext].values
			neighbor_df[cate_features_ext] = cate_neighbor
			scale_neighbor_df[cate_features_ext] = cate_neighbor
			scale[idx_ext : idx_ext + len(cate_features_ext)] = 1
			mean[idx_ext : idx_ext + len(cate_features_ext)] = 0

			scale_neighbor_df.loc[~(scale_neighbor_df[cate_features_ext] == data[cate_features_ext].values).all(axis='columns'), cate_features_ext] = 0
			idx_ext += len(cate_features_ext)

	neighbor_df.loc[0] = data.values
	scale_neighbor_df.loc[0] = data.values
	scale_neighbor_df = (scale_neighbor_df - mean) / scale

	return neighbor_df, scale_neighbor_df

def SurvLIME(explainer, data, n_nearest=100, id=None):
	"""
	Compute SurvLIME


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model

	data :  `np.ndarray`, shape=(n_samples, n_features)
		New observations for which predictions need to be explained

	n_nearest : `int`, default = 100
		Number of neighbor points of the interest observations to be generated

	Returns
	-------
	SurvLIME_df : `pd.Dataframe`
		The SurvLIME results
	"""

	def objective(x, model_chf, baseline_chf, neibourg_points, distance_weights, straightening_weights, unique_times):

		dt = unique_times[1:] - unique_times[:-1]
		tmp = np.sum((straightening_weights[:, :-1] ** 2) * (((np.log(model_chf[:, :-1]) - np.log(baseline_chf[:-1]) - neibourg_points.dot(x).reshape((-1, 1))) ** 2) * dt), axis=1)
		f_x = np.sum(distance_weights * tmp)

		return f_x

	sup = 1 + 1e-4
	label = explainer.label
	surv_time = label[:, 0]
	surv_ind = label[:, 1].astype('bool')
	# cumulative hazard function
	unique_times, baseline_chf = nelson_aalen_estimator(surv_ind, surv_time)
	baseline_chf = baseline_chf + sup

	# TODO: Update for all individuals
	if id is None:
		id = 0
	interest_point = data.iloc[[id]]
	n_feats = data.shape[1]
	neighbor_df, scale_neighbor_df = neighbor_generation(explainer, interest_point, n_nearest)
	neibourg_points = neighbor_df.values
	scale_neighbor = scale_neighbor_df.values

	model_chf = predict(explainer, neibourg_points, unique_times, type="chf")
	model_chf = model_chf + sup
	distances = np.linalg.norm(scale_neighbor - scale_neighbor[0], axis=1)
	kernel_width = np.sqrt(n_feats) * 0.75
	distance_weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
	straightening_weights = model_chf / np.log(model_chf)

	init_point = np.zeros(n_feats)
	args = (model_chf, baseline_chf, neibourg_points, distance_weights, straightening_weights, unique_times)
	coefs = minimize(objective, init_point, args=args, method='BFGS', options={'gtol': 1e-8})["x"]

	SurvLIME_res = np.stack([data.columns.values, coefs]).T
	SurvLIME_df = pd.DataFrame(data = SurvLIME_res, columns=["feats", "coefs"]).reset_index(drop=True)
	SurvLIME_df.coefs = SurvLIME_df.coefs.values.astype(float)
	SurvLIME_df["FI"] = SurvLIME_df.coefs.values * interest_point.values.flatten()

	return SurvLIME_df

def plot_SurvLIME(res, id=0):
	"""
	Visualize the SurvLIME results

	Parameters
	----------
	res : `pd.Dataframe`
		SurvLIME result to be visualize
	"""

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	colors = ['g' if c >= 0 else 'r' for c in res.coefs.values]
	sns.barplot(data=res, y="feats", x="FI", palette=colors)

	plt.xlabel("Local importance")
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