import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')

from src.local_explaination import individual_conditional_expectation
from src import performance
from src.prediction import predict

def partial_dependence_plots(explainer, selected_features, n_sel_samples = 100,
                                       n_grid_points = 50, type = "survival"):
	"""
    Compute partial dependence plot (PDP)


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
    DPD_df : `np.ndarray`, shape=()
        Returns the PDP value of selected features
	"""

	ICE_df = individual_conditional_expectation(explainer, selected_features,
	                                   n_sel_samples, n_grid_points, type)

	PDP_df = ICE_df.groupby([selected_features, 'times']).mean().reset_index()[[selected_features, "times", "pred"]]

	return PDP_df


def plot_PDP(explainer, res, explained_feature = ""):
	"""
	Visualize the PDP results

	Parameters
    ----------
    res : `pd.Dataframe`
		PDP result to be visualize
	"""

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	if explained_feature in explainer.numeric_feats:
		X_unique = np.unique(res[explained_feature].values)
		n_unique = len(X_unique)
		X_norm = (X_unique - min(X_unique)) / (max(X_unique) - min(X_unique))
		cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(X_unique), True), cmap='BrBG')
		for i in np.arange(0, n_unique):
			res_i = res.loc[(res[explained_feature] == X_unique[i])]
			sns.lineplot(data=res_i, x="times", y="pred", color=cmap.get_cmap()(X_norm[i]))

		plt.colorbar(cmap, orientation='vertical', label=explained_feature, ax=ax)
	else:
		res_sorted = res.sort_values(by=explained_feature)
		sns.lineplot(data=res_sorted, x="times", y="pred", hue = explained_feature)

	ax.set_ylim(0, 1)
	plt.xlabel("Time")
	plt.ylabel("Survival prediction")
	plt.title("DPD for feature {0}".format(explained_feature))
	plt.savefig("DPD_feature_{0}.pdf".format(explained_feature), bbox_inches='tight')
	plt.show()


def permutation_feature_importance(explainer, feats, surv_labels, eval_times=None,
                                   n_perm = 10, loss="brier_score", type="ratio"):
	"""
	Compute permutation feature importance (PFI)


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model

	Returns
	-------
	PFI_df : `np.ndarray`, shape=()
		Returns the PFI value of selected features
	"""

	if loss == "brier_score":
		bs_perf = performance.evaluate(explainer, feats, surv_labels,
		                               times=eval_times, metric="brier_score")["perf"].values

		feats_name = feats.columns.values.tolist()
		feat_importance_df_cols = ["feat", "times", "perf"]
		feat_importance_df = pd.DataFrame(columns=feat_importance_df_cols)
		if eval_times is None:
			eval_times = explainer.times
		n_eval_times  = len(eval_times)

		feats_name_ext = explainer.numeric_feats.copy()
		if explainer.cate_feats is not None:
			feats_name_org = explainer.numeric_feats + explainer.cate_feats
			for cate_feat_name in explainer.cate_feats:
				cate_feat_name_list = []
				for feat_name in feats_name:
					if cate_feat_name in feat_name:
						cate_feat_name_list.append(feat_name)
				if len(cate_feat_name_list):
					feats_name_ext.append(cate_feat_name_list)
		else:
			feats_name_org = explainer.numeric_feats

		for i in range(len(feats_name_ext)):
			feat_name = feats_name_ext[i]
			feat_name_org = feats_name_org[i]
			bs_perf_perm = np.zeros(n_eval_times)
			for k in range(n_perm):
				feat_perm = feats.copy(deep=True)[feat_name].values
				np.random.shuffle(feat_perm)
				feats_perm_df = feats.copy(deep=True)
				feats_perm_df[feat_name] = feat_perm
				tmp = performance.evaluate(explainer, feats_perm_df, surv_labels,
				                           times=eval_times, metric="brier_score")["perf"].values
				bs_perf_perm += (1 / n_perm) * tmp

			if type == "ratio":

				importance_ratio = bs_perf / bs_perf_perm
				importance_ratio_data = np.stack(([feat_name_org] * n_eval_times, eval_times, importance_ratio)).T
				additional_ratio = pd.DataFrame(data=importance_ratio_data, columns=feat_importance_df_cols)
				feat_importance_df = pd.concat([feat_importance_df, additional_ratio], ignore_index=False)

			if type == "loss":
				bs_loss_data = np.stack(([feat_name_org] * n_eval_times, eval_times, bs_perf_perm)).T
				additional_loss = pd.DataFrame(data=bs_loss_data, columns=feat_importance_df_cols)
				feat_importance_df = pd.concat([feat_importance_df, additional_loss], ignore_index=False)

		if type == "loss":
			bs_loss_data = np.stack((["full_model"] * n_eval_times, eval_times, bs_perf)).T
			additional_loss = pd.DataFrame(data=bs_loss_data, columns=feat_importance_df_cols)
			feat_importance_df = pd.concat([feat_importance_df, additional_loss], ignore_index=False)

	feat_importance_df[["times", "perf"]] = feat_importance_df[["times", "perf"]].apply(pd.to_numeric)

	return feat_importance_df

def plot_PFI(res, type, legend_loc='lower right'):
	"""
	Visualize the PFI results

	Parameters
	----------
	res : `pd.Dataframe`
		PFI result to be visualize
	"""
	feats_names = np.unique(res.feat.values)

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	for feat_name in feats_names:
		res_feat = res.loc[(res.feat == feat_name)]
		sns.lineplot(data=res_feat, x="times", y="perf", label=feat_name)

	plt.xlabel("Times")
	if type == "loss":
		plt.ylabel("Brier score loss after permutation")
	if type == "ratio":
		plt.ylabel("Brier score ratio after permutation")

	plt.legend(loc=legend_loc, ncol=2, prop = {"size": 12})
	plt.title("Permutation feature importance")
	plt.savefig("Permutation_feature_importance.pdf")
	plt.show()

def accumulated_local_effects_plots(explainer, selected_features, type = "survival"):
	"""
	Compute accumulated local effects plots (ALE)

	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	data = explainer.data.copy(deep=True).sort_values(by=[selected_features])
	var_values = data[selected_features].values
	qt_list = np.arange(0., 1.01, 0.1)
	grid_qt_values = np.quantile(var_values, qt_list)
	var_values_idx = [min(np.abs(grid_qt_values - var).argmin(), len(qt_list) - 2) for var in var_values]
	var_lower = np.array([grid_qt_values[i] for i in var_values_idx])
	var_upper = np.array([grid_qt_values[i + 1] for i in var_values_idx])
	data_lower, data_upper = data.copy(deep=True), data.copy(deep=True)
	data_lower[selected_features] = var_lower
	data_upper[selected_features] = var_upper

	eval_times = np.unique(explainer.label[:, 0])[10::100]
	if type == "survival":
		lower_pred = predict(explainer, data_lower, eval_times)
		upper_pred = predict(explainer, data_upper, eval_times)
	elif type == "chf":
		lower_pred = predict(explainer, data_lower, eval_times, "chf")
		upper_pred = predict(explainer, data_upper, eval_times, "chf")
	else:
		raise ValueError("Unsupported")

	n_times = len(eval_times)
	groups = np.repeat(var_values_idx, n_times)
	diff_pred = upper_pred.copy(deep=True)[["pred", "times"]]
	diff_pred["pred"] = diff_pred["pred"].values - lower_pred.copy(deep=True)["pred"].values
	diff_pred["groups"] = groups

	ALE_df = diff_pred.groupby(['groups', 'times']).mean().reset_index()[["groups", "times", "pred"]]
	group_values = np.repeat(grid_qt_values[:-1], n_times)
	ALE_df["group_values"] = group_values

	return ALE_df

def plot_ALE(res, explained_feature=None):
	"""
    Visualize the ALE results

    Parameters
    ----------
    res : `pd.Dataframe`
        PFI result to be visualize
    explained_feature : `str`
        Name of explained feature
    """
	groups = np.unique(res.groups.values)
	group_values = np.unique(res.group_values.values)

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	group_values_norm = (group_values - min(group_values)) / (max(group_values) - min(group_values))
	n_groups= len(groups)
	cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(group_values), True), cmap='BrBG')
	for i in range(n_groups):
		group = groups[i]
		res_group = res.loc[(res.groups <= group)].groupby(['times']).sum().reset_index()[["groups", "times", "pred"]]
		sns.lineplot(data=res_group, x="times", y="pred", color=cmap.get_cmap()(group_values_norm[i]))

	plt.xlabel("Time")
	plt.ylabel("")
	plt.colorbar(cmap, orientation='vertical', label=explained_feature, ax=ax, pad=0.1)
	plt.title("Accumulated local effects")
	plt.show()

def feature_interaction(explainer):
	"""
	Compute feature interaction

	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	raise ValueError("Not supported yet")


def functional_decomposition(explainer):
	"""
	Compute functional decomposition


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	raise ValueError("Not supported yet")