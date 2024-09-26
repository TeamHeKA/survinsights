import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')

from src.local_explaination import individual_conditional_expectation
from src import performance

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

	PDP_df = ICE_df.groupby(['X', 'times']).mean().reset_index()[["X", "times", "pred"]]

	return PDP_df

def plot_PDP(res):
	"""
	Visualize the PDP results

	Parameters
	----------
	res : `pd.Dataframe`
		PDP result to be visualize
	"""
	pred_times = np.unique(res.times.values)
	n_pred_times = len(pred_times)

	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	times_norm = (pred_times - min(pred_times)) / (
				max(pred_times) - min(pred_times))
	cmap = mpl.cm.ScalarMappable(
		norm=mpl.colors.Normalize(0.0, max(pred_times), True), cmap='BrBG')
	for i in np.arange(0, n_pred_times):
		res_i = res.loc[(res.times == pred_times[i])]
		sns.lineplot(data=res_i, x="X", y="pred",
		             color=cmap.get_cmap()(times_norm[i]))

	ax.set_ylim(0, 1)
	plt.xlabel("")
	plt.ylabel("Survival prediction")
	plt.title("DPD for feature x0")
	plt.colorbar(cmap, orientation='horizontal', label='Time', ax=ax)
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
		n_eval_times  = len(eval_times)
		for feat_name in feats_name:
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
				importance_ratio_data = np.stack(([feat_name] * n_eval_times, eval_times, importance_ratio)).T
				additional_ratio = pd.DataFrame(data=importance_ratio_data, columns=feat_importance_df_cols)
				feat_importance_df = pd.concat([feat_importance_df, additional_ratio], ignore_index=False)

	feat_importance_df[["times", "perf"]] = feat_importance_df[["times", "perf"]].apply(pd.to_numeric)

	return feat_importance_df

def plot_PFI(res):
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
	plt.ylabel("")
	plt.legend(loc='lower left', ncol=3)
	plt.title("Permutation feature importance")
	plt.show()