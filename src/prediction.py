import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')


def predict(explainer, data, times=None, type="survival"):
	"""
	Calculate model prediction in a unified way


	Parameters
    ----------
    explainer :  `class`
        A Python class used to explain the survival model

    data :  `np.ndarray`, shape=(n_samples, n_features)
        Data used for the prediction

    times :  `np.ndarray`, shape=(n_times,), default = None
		An array of times for the desired prediction to be evaluated at

    type : `str`, default = "survival"
        The character of output type, either "risk", "survival" or "chf" depending
        on the desired output


    Returns
    -------
    pred : `np.ndarray`, shape=(n_samples, n_times)
        The matrix contains the prediction
	"""

	n_samples = data.shape[0]

	if times is None:
		times = explainer.times
	else:
		times = np.unique(times)
	n_times = len(times)

	if type == "survival":
		preds = explainer.sf(data)
	elif type == "chf":
		preds = explainer.chf(data)
	else:
		raise ValueError("Unsupported type")

	if isinstance(data, pd.DataFrame):
		pred_df = pd.DataFrame(columns=["id", "times", "pred"])
		for i in range(n_samples):
			for j in range(n_times):
				time_j = times[j]
				pred_df.loc[len(pred_df)] = [i, time_j, preds[i](time_j)]
				#surv_pred_i = preds[i].y
				#time_pred_i = preds[i].x
				#idx_time_ij = (np.abs(time_pred_i - time_j)).argmin()
				#pred_ij = surv_pred_i[idx_time_ij]
				#pred_df.loc[len(pred_df)] = [i, time_j, pred_ij]

		return pred_df

	else:
		pred = np.zeros((n_samples, n_times))
		for j in range(n_times):
			time_j = times[j]
			for i in range(n_samples):
				pred[i, j] = preds[i](time_j)
			#surv_pred_i = preds[i].y
				#time_pred_i = preds[i].x
				#idx_time_ij = (np.abs(time_pred_i - time_j)).argmin()
				#pred[i, j] = surv_pred_i[idx_time_ij]

		return pred


def plot_prediction(pred, type):
	"""
	Plot the prediction of survival model


	Parameters
	----------

	pred :  `pd.Dataframe`, shape=(n_samples, n_times)
		A dataframe contains the prediction values

	type : `str`
        The character of output type, either "risk", "survival" or "chf" depending
        on the desired output

	"""
	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	sns.lineplot(data=pred, x="times", y="pred", hue="id")
	ax.get_legend().remove()
	ax.set_xlim(min(pred.times.values), max(pred.times.values))
	plt.xlabel("Times", fontsize=20)
	if type == "survival":
		ax.set_ylim(0, 1)
		plt.ylabel("Survival function", fontsize=20)
	elif type == "chf":
		plt.ylabel("Cumulative hazard function", fontsize=20)
	elif type == "risk":
		plt.ylabel("Hazard function", fontsize=20)
	else:
		raise ValueError("Only support output type survival, chf, risk")

	plt.show()