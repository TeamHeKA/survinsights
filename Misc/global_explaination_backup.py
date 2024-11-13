import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')
from itertools import combinations

from src.local_explaination import individual_conditional_expectation, individual_conditional_expectation_2d
from src import performance
from src.prediction import predict
from src.utils import feat_order

def partial_dependence_plots(explainer, explained_feature_name, n_sel_samples = 100,
                                       n_grid_points = 50, output_type = "survival"):
	"""
    Compute partial dependence plot (PDP)


	Parameters
	----------
	explainer : object
		A Python class instance to explain the survival model.
	explained_feature_name : str
		The name of the feature to generate the PDP for.
	n_sel_samples : int, optional, default=100
		Number of samples for calculating aggregated profiles.
	n_grid_points : int, optional, default=50
		Number of grid points for calculating aggregated profiles.
	output_type : str, optional, default="survival"
		Type of output to generate - options are "risk", "survival", or "chf".

    Returns
    -------
    DataFrame containing the PDP values for the selected feature.
	"""

	ICE_df = individual_conditional_expectation(explainer, explained_feature_name,
												n_sel_samples, n_grid_points, output_type)

	PDP_df = ICE_df.groupby([explained_feature_name, 'times']).mean().reset_index()[[explained_feature_name, "times", "pred"]]

	return PDP_df


def plot_PDP(explainer, results, explained_feature_name = ""):
	"""
	Visualize the Partial Dependence Plot (PDP) results.

	Parameters
	----------
	explainer : object
	    An explainer object containing feature information.
	results : pd.DataFrame
	    A dataframe containing the PDP results to visualize.
	explained_feature_name : str, optional, default=""
	    Name of the feature for PDP visualization.
	"""

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	if explained_feature_name in explainer.numeric_feat_names:
		unique_values = np.unique(results[explained_feature_name].values)
		normalized_values = (unique_values - min(unique_values)) / (max(unique_values) - min(unique_values))
		cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(unique_values), True), cmap='BrBG')
		for i, value in enumerate(unique_values):
			subset = results[results[explained_feature_name] == value]
			sns.lineplot(data=subset, x="times", y="pred", color=cmap.get_cmap()(normalized_values[i]), ax=ax)

		plt.colorbar(cmap, ax=ax, orientation='vertical', label=explained_feature_name)
	else:
		res_sorted = results.sort_values(by=explained_feature_name)
		sns.lineplot(data=res_sorted, x="times", y="pred", hue = explained_feature_name, ax=ax)

	ax.set_ylim(0, 1)
	plt.xlabel("Time")
	plt.ylabel("Survival prediction")
	plt.title(f"PDP for feature {explained_feature_name}")
	plt.savefig(f"PDP_feature_{explained_feature_name}.pdf", bbox_inches='tight')
	plt.show()


def permutation_feature_importance(explainer, features, surv_labels, eval_times=None,
                                   n_perm = 10, loss="brier_score", output_type="ratio"):
	"""
	Compute permutation feature importance (PFI) for survival models.

	Parameters
	----------
	explainer : object
	    An explainer class instance for the survival model.
	features : pd.DataFrame
	    DataFrame of features to be permuted.
	surv_labels : pd.DataFrame
	    Survival labels for evaluating the model performance.
	eval_times : list or np.array, optional
	    Times at which to evaluate performance.
	n_perm : int, optional, default=10
	    Number of permutations to perform.
	loss : str, optional, default="brier_score"
	    Performance metric for evaluation.
	output_type : str, optional, default="ratio"
	    Type of PFI - "ratio" or "loss".

	Returns
	-------
	pd.DataFrame
	    DataFrame containing the PFI values for the selected features.
	"""

	eval_times = eval_times or explainer.times
	# Compute the baseline performance of the model.
	base_performance = performance.evaluate(explainer, features, surv_labels, times=eval_times, metric="brier_score")["perf"].values
	expanded_feature_names = expand_feature_names(explainer, features.columns.tolist())

	feat_importance_df = pd.DataFrame(columns=["feat", "times", "perf"])
	for feature_group in expanded_feature_names:
		original_name = feature_group[0] if isinstance(feature_group, list) else feature_group

		permuted_perf = get_permuted_performance(
			explainer, features, surv_labels, eval_times, feature_group, n_perm, loss
		)

		feat_importance_df = update_feature_importance_df(
			feat_importance_df, original_name, eval_times, base_performance, permuted_perf, output_type
		)

	if output_type == "loss":
		feat_importance_df = add_full_model_performance(feat_importance_df, eval_times, base_performance)

	feat_importance_df[["times", "perf"]] = feat_importance_df[["times", "perf"]].apply(pd.to_numeric)

	return feat_importance_df

def expand_feature_names(explainer, features_names):
	"""
    Expand categorical features for permutation.

    Parameters
    ----------
    explainer : object
        An explainer instance containing feature information.
    features_names : list of str
        List of feature names.

    Returns
    -------
    expanded_names : list of str
        List of expanded feature names with categorical features grouped.
    """

	expanded_names = explainer.numeric_feat_names.copy()
	if explainer.cate_feat_names:
		expanded_names += [[cate_name] +
			[name for name in features_names if cate_name in name]
			for cate_name in explainer.cate_feat_names if any(cate_name in f for f in features_names)
		]
	return expanded_names

def get_permuted_performance(explainer, features, surv_labels, eval_times, feature_group, n_perm, metric):
	"""
	Compute performance after permuting feature(s).

	Parameters
	----------
	explainer : object
		Explainer instance.
	features : pd.DataFrame
		Dataframe containing features.
	surv_labels : pd.DataFrame
		Survival labels for evaluation.
	eval_times : list or np.array
		Evaluation times.
	feature_group : list or str
		Feature(s) to be permuted.
	n_perm : int
		Number of permutations.
	metric : str
		Metric for evaluation.

	Returns
	-------
	np.array
		Array of permuted performance values.
	"""
	permuted_perf = np.zeros(len(eval_times))
	for _ in range(n_perm):
		permuted_features = permute_feature(features, feature_group)
		permuted_perf += performance.evaluate(
				explainer, permuted_features, surv_labels, times=eval_times, metric=metric)["perf"].values / n_perm
	return permuted_perf

def permute_feature(features, feature_names):
	"""
	Permute values for specified feature(s) in the features DataFrame.

	Parameters
	----------
	features : pd.DataFrame
		The feature data.
	feature_names : list or str
		Feature(s) to be permuted.

	Returns
	-------
	pd.DataFrame
		DataFrame with the specified feature(s) permuted.
	"""
	permuted_features = features.copy()
	if isinstance(feature_names, list):
		permuted_features[feature_names[1:]] = np.random.permutation(permuted_features[feature_names[1:]])
	else:
		permuted_features[feature_names] = np.random.permutation(permuted_features[feature_names])
	return permuted_features

def update_feature_importance_df(feat_importance_df, feat_name, eval_times, base_perf, permuted_perf, output_type):
	"""
	Update the feature importance DataFrame with computed values.

	Parameters
	----------
	feat_importance_df : pd.DataFrame
		DataFrame to update.
	feat_name : str
		Name of the feature.
	eval_times : list or np.array
		Evaluation times.
	base_perf : np.array
		Base performance values.
	permuted_perf : np.array
		Permuted performance values.
	output_type : str
		Type of feature importance ("ratio" or "loss").

	Returns
	-------
	pd.DataFrame
		Updated feature importance DataFrame.
	"""
	if output_type == "ratio":
		importance_feat = np.stack(([feat_name] * len(eval_times), eval_times, base_perf / permuted_perf)).T
	else:
		importance_feat = np.stack(([feat_name] * len(eval_times), eval_times, permuted_perf)).T

	return pd.concat([feat_importance_df, pd.DataFrame(importance_feat, columns=feat_importance_df.columns)], ignore_index=True)

def add_full_model_performance(feat_importance_df, eval_times, base_performance):
	"""
	Add full model performance to the feature importance DataFrame for 'loss' output type.

	Parameters
	----------
	feat_importance_df : pd.DataFrame
		The feature importance DataFrame to update.
	eval_times : list or np.array
		Evaluation times.
	base_performance : np.array
		Base performance values.

	Returns
	-------
	pd.DataFrame
		Updated feature importance DataFrame with full model performance.
	"""
	full_model_feat = np.stack((["full_model"] * len(eval_times), eval_times, base_performance)).T
	return pd.concat([feat_importance_df, pd.DataFrame(full_model_feat, columns=feat_importance_df.columns)], ignore_index=True)


def plot_PFI(results, output_type, legend_loc='lower right'):
	"""
	Visualize the Permutation Feature Importance (PFI) results.

	Parameters
	----------
	results : pd.DataFrame
	    DataFrame containing the PFI results.
	output_type : str
	    The type of importance displayed - "loss" or "ratio".
	legend_loc : str, optional
	    Location of the legend on the plot.
	"""
	feature_names = results.feat.unique()
	fig, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	for feat_name in feature_names:
		sns.lineplot(data=results[results.feat == feat_name], x="times", y="perf", label=feat_name)

	plt.xlabel("Times")
	plt.ylabel("Brier Score Ratio" if output_type == "ratio" else "Brier Score Loss")
	plt.legend(loc=legend_loc, ncol=2, prop={"size": 12})
	plt.title("Permutation Feature Importance")
	plt.savefig("Permutation_feature_importance.pdf")
	plt.show()

def accumulated_local_effects_plots(explainer, explained_feature_name, output_type = "survival"):
	"""
	Compute accumulated local effects plots (ALE) for survival models.

	Parameters
	----------
	explainer : object
	    An explainer class instance for the survival model.
	explained_feature_name : str
	    The feature for which ALE should be computed.
	output_type : str, optional, default="survival"
	    Type of prediction - "survival" or "chf".

	Returns
	-------
	pd.DataFrame
	    DataFrame containing ALE values for the selected feature.
	"""

	if explained_feature_name in explainer.numeric_feat_names:
		return compute_numeric_ale(explainer, explained_feature_name, output_type)
	else:
		return compute_categorical_ale(explainer, explained_feature_name, output_type)


def compute_numeric_ale(explainer, feat_name, output_type):
	"""
    Compute ALE for a numeric feature by dividing it into quantile-based intervals.

    Parameters
    ----------
    explainer : object
        The explainer instance with data and survival labels.
    feature : str
        The numeric feature for which ALE is computed.
    output_type : str
        The type of prediction output ("survival" or "chf").

    Returns
    -------
    pd.DataFrame
        DataFrame containing ALE values for the feature at different quantile levels and times.
    """
	features = explainer.features.copy(deep=True).sort_values(by=[feat_name])
	grid_values = np.quantile(features[feat_name].values, np.arange(0., 1.01, 0.1))
	value_group = [np.abs(grid_values[:-1] - val).argmin() for val in features[feat_name]]

	data_lower, data_upper = create_bound_datasets(features, feat_name, grid_values)
	eval_times = np.unique(explainer.survival_labels[:, 0])

	if output_type in ["survival", "chf"]:
		lower_pred = predict(explainer, data_lower, eval_times, output_type)
		upper_pred = predict(explainer, data_upper, eval_times, output_type)
	else:
		raise ValueError("Unsupported")

	ale_diff_df = calculate_ale_diff(lower_pred, upper_pred, value_group, eval_times)

	return finalize_ale(ale_diff_df, feat_name, explainer, grid_values, eval_times)

def compute_categorical_ale(explainer, feat_name, output_type):
	"""
	Compute ALE for a categorical feature by shifting between adjacent categories.

	Parameters
	----------
	explainer : object
		The explainer instance with data and survival labels.
	feat_name : str
		The categorical feature for which ALE is computed.
	output_type : str
		The type of prediction output ("survival" or "chf").

	Returns
	-------
	pd.DataFrame
		DataFrame containing ALE values for the feature at different categories and times.
	"""
	features = explainer.features.copy(deep=True)
	category_order = feat_order(explainer, feat_name)
	feat_name_ext = [feat_col for feat_col in features.columns.values if feat_name in feat_col]
	value_group = [category_order.tolist().index(val.tolist()) for val in features[feat_name_ext].values]
	eval_times = np.unique(explainer.survival_labels[:, 0])
	if output_type in ["survival", "chf"]:
		inc_group_sel = [group > 0 for group in value_group]
		dec_group_sel = [group < len(category_order) - 1 for group in value_group]
		feat_inc, feat_dec = features.copy(deep=True)[inc_group_sel], features.copy(deep=True)[dec_group_sel]
		feat_inc[feat_name_ext] = [category_order[group - 1] for group in value_group if group > 0]
		feat_dec[feat_name_ext] = [category_order[group + 1] for group in value_group if group < len(category_order) - 1]
		inc_pred = predict(explainer, feat_inc, eval_times, output_type)[["id", "pred", "times"]]
		dec_pred = predict(explainer, feat_dec, eval_times, output_type)[["id", "pred", "times"]]
		org_inc_pred = predict(explainer, features[inc_group_sel], eval_times)[["id", "pred", "times"]]
		org_dec_pred = predict(explainer, features[dec_group_sel], eval_times)[["id", "pred", "times"]]

		n_times = len(eval_times)
		inc_group = [group for group in value_group if group > 0]
		inc_groups_ext = np.repeat(inc_group, n_times)
		dec_group = [group + 1 for group in value_group if group < len(category_order) - 1]
		dec_groups_ext = np.repeat(dec_group, n_times)
		inc_pred["pred"] = org_inc_pred["pred"].values - inc_pred["pred"].values
		dec_pred["pred"] = dec_pred["pred"].values - org_dec_pred["pred"].values
		inc_pred["groups"] = inc_groups_ext
		dec_pred["groups"] = dec_groups_ext
		ale_diff_df = pd.concat([inc_pred, dec_pred], ignore_index=True)
	else:
		raise ValueError("Unsupported output type")


	return finalize_ale(ale_diff_df, feat_name, explainer, category_order, eval_times)


def create_bound_datasets(features, feat_name, grid_values):
	"""
	Create datasets with feature values shifted to lower and upper quantile boundaries.

	Parameters
	----------
	data : pd.DataFrame
		Original data for ALE computation.
	feature : str
		The feature to be shifted.
	grid_values : np.array
		Quantile-based grid values for the feature.

	Returns
	-------
	tuple of pd.DataFrame
		Two DataFrames with the feature values set to the lower and upper quantile boundaries.
	"""
	values_idx = [np.abs(grid_values[:-1] - val).argmin() for val in features[feat_name]]
	feat_lower, feat_upper = features.copy(deep=True), features.copy(deep=True)
	feat_lower[feat_name], feat_upper[feat_name] = (np.array([grid_values[i] for i in values_idx]),
													np.array([grid_values[i + 1] for i in values_idx]))
	return feat_lower, feat_upper

def calculate_ale_diff(lower_pred, upper_pred, value_group, eval_times):
	"""
	Calculate ALE differences between predictions with feature values at lower and upper quantile bounds.

	Parameters
	----------
	lower_pred : pd.DataFrame
		Predictions for the lower quantile-bound dataset.
	upper_pred : pd.DataFrame
		Predictions for the upper quantile-bound dataset.
	grid_values : np.ndarray
		Quantile grid values.
	eval_times : np.ndarray
		Evaluation times for predictions.

	Returns
	-------
	pd.DataFrame
		DataFrame of ALE differences with associated groups.
	"""
	n_times = len(eval_times)
	groups_ext = np.repeat(value_group, n_times)
	diff_pred = upper_pred[["id", "pred", "times"]].copy()
	diff_pred["pred"] -= lower_pred["pred"].values
	diff_pred["groups"] = groups_ext
	return diff_pred


def finalize_ale(ale_diff_df, feat_name, explainer, grid_values, eval_times):
	"""
    Finalize ALE computation by grouping and centering values over time.

    Parameters
    ----------
    ale_diff_df : pd.DataFrame
        DataFrame of ALE differences.
    grid_values : np.ndarray
        Quantile grid values for the feature.
    eval_times : np.ndarray
        Times for ALE evaluation.

    Returns
    -------
    pd.DataFrame
        DataFrame with finalized ALE values and centered effects.
    """
	ALE_group_df = ale_diff_df.groupby(['groups', 'times']).mean().reset_index()[["groups", "times", "pred"]]
	n_times = len(eval_times)
	if feat_name in explainer.cate_feat_names:
		tmp_df = pd.DataFrame(data=np.array([np.zeros(n_times), eval_times, np.zeros(n_times)]).T,
							  columns=["groups", "times", "pred"])
		ALE_group_df = pd.concat([tmp_df, ALE_group_df])
	groups_unique = np.unique(ALE_group_df["groups"].values)
	n_groups = len(groups_unique)
	ALE_df = pd.DataFrame(columns=["groups", "times", "pred"])
	for i in range(n_groups):
		group = groups_unique[i]
		res_group = ALE_group_df.loc[(ALE_group_df.groups <= group)].groupby(['times']).sum().reset_index()[["groups", "times", "pred"]]
		res_group.groups = group
		ALE_df = pd.concat([ALE_df, res_group], ignore_index=True)

	if feat_name in explainer.numeric_feat_names:
		group_values = np.repeat(grid_values[:-1], n_times)
	else:
		encoder = explainer.encoders[feat_name]
		group_values_ = encoder.inverse_transform(grid_values).flatten()
		group_values = np.repeat(group_values_, n_times)

	id_df = ale_diff_df[["id", "groups"]].drop_duplicates()
	ALE_df_ext = ALE_df.join(id_df.set_index('groups'), on='groups')
	ALE_df_mean = ALE_df_ext.groupby(['times']).mean().reset_index()[["times", "pred"]]
	ALE_df_mean = ALE_df_mean.rename(columns={"pred": "pred_mean"})
	ALEc_df = ALE_df.join(ALE_df_mean[["times", "pred_mean"]].set_index('times'), on='times')
	ALEc_df["alec"] = ALEc_df.pred.values - ALEc_df.pred_mean.values
	ALEc_df["group_values"] = group_values

	return ALEc_df

def plot_ALE(explainer, res, explained_feature):
	"""
    Visualize the ALE results

    Parameters
    ----------
    res : `pd.Dataframe`
        ALE result to be visualized
    explained_feature : `str`
        Name of explained feature
    """

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]

	if explained_feature in explainer.numeric_feat_names:
		groups = np.unique(res.groups.values)
		group_values = np.unique(res.group_values.values)
		group_values_norm = (group_values - min(group_values)) / (max(group_values) - min(group_values))
		n_groups= len(groups)
		cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(group_values), True), cmap='BrBG')
		for i in range(n_groups):
			group = groups[i]
			sns.lineplot(data=res.loc[(res.groups == group)], x="times", y="alec",
						 color=cmap.get_cmap()(group_values_norm[i]))
		plt.colorbar(cmap, orientation='vertical', label=explained_feature, ax=ax, pad=0.1)
	else:
		sns.lineplot(data=res, x="times", y="alec", hue="group_values", ax=ax)
		plt.legend(prop={"size": 12})


	plt.xlabel("Time")
	plt.ylabel("")
	plt.title("Accumulated local effects")
	plt.show()


def feature_interaction(explainer, explained_features=None, n_sel_samples=10, n_grid_points=10):
	"""
	Compute feature interaction

	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	feats = explainer.numeric_feat_names + explainer.cate_feat_names
	if explained_features == None:
		explained_features_list = combinations(feats, 2)
		explained_features_list = [list(i) for i in explained_features_list]
	else:
		explained_features_list = [[explained_features, feat] for feat in feats if feat != explained_features]

	H_stat_df_list = []
	for explained_feature in explained_features_list:
		ICE_df = individual_conditional_expectation_2d(explainer, explained_feature, n_sel_samples, n_grid_points)
		pdp_cols = ["times"] + explained_feature
		PDP_df_merged = ICE_df.groupby(pdp_cols).mean().reset_index()[pdp_cols + ["pred"]]

		for i in range(len(explained_feature)):
			feature = explained_feature[i]
			ICE_df_feat = individual_conditional_expectation(explainer, feature, n_sel_samples, n_grid_points=None)
			pdp_cols_feat = ["times", feature]
			PDP_df_feat = ICE_df_feat.groupby(pdp_cols_feat).mean().reset_index()[pdp_cols_feat + ["pred"]]
			PDP_df_feat = PDP_df_feat.rename(columns={"pred": "pred_{}".format(i + 1)})
			PDP_df_merged = PDP_df_merged.merge(PDP_df_feat, how='inner', on=['times', feature])

		PDP_df_merged["var"] = (PDP_df_merged.pred.values - PDP_df_merged.pred_1.values - PDP_df_merged.pred_2.values) ** 2
		PDP_df_merged["cor_sq"] = (PDP_df_merged.pred.values) ** 2
		tmp_df_1 = PDP_df_merged[["times", "var"]].groupby("times").sum().reset_index()[["times", "var"]]
		tmp_df_2 = PDP_df_merged[["times", "cor_sq"]].groupby("times").sum().reset_index()[["times", "cor_sq"]]
		H_stat_df = tmp_df_1[["times"]]
		H_stat_df["H_stat"] = tmp_df_1["var"].values / tmp_df_2.cor_sq.values
		H_stat_df[["feat_1", "feat_2"]] = explained_feature
		H_stat_df_list.append(H_stat_df)
	H_stat_df_final = pd.concat(H_stat_df_list)

	return H_stat_df_final

def plot_feature_interaction(H_stat_df):
	"""
    Visualize the feature interaction results

    Parameters
    ----------
    H_stat_df : `pd.Dataframe`
        H statistics result to be visualize
    """
	pair_list = H_stat_df[["feat_1", "feat_2"]].drop_duplicates().values

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	for pair in pair_list:
		H_stat_df_pair = H_stat_df[(H_stat_df[["feat_1", "feat_2"]].values == pair).all(axis=1)]
		sns.lineplot(data=H_stat_df_pair, x="times", y="H_stat", label=pair[0] + "+" + pair[1])

	plt.xlabel("Time")
	plt.ylabel("")
	plt.legend(prop = {"size": 12})
	plt.title("Feature interaction")
	plt.show()

def functional_decomposition(explainer):
	"""
	Compute functional decomposition


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model
	"""

	raise ValueError("Not supported yet")