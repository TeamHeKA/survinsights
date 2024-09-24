import numpy as np
from src.local_explaination import individual_conditional_expectation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')

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
	plt.colorbar(cmap, orientation='horizontal', label='Time')
	plt.show()