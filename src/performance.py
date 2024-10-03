import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, brier_score, cumulative_dynamic_auc
from src.prediction import predict
from src.utils import convert_surv_label_structarray
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')

def evaluate(explainer, data, label, times=None, metric="brier_score"):
	"""
	Calculate survival model performance based on the desired metric


	Parameters
	----------
	explainer :  `class`
		A Python class used to explain the survival model

	data :  `np.ndarray`, shape=(n_samples, n_feattures)
		Data used for the prediction

	times :  `np.ndarray`, shape=(n_times,), default = None
		An array of times for the desired prediction to be evaluated at

	metric : `str`, default = "brier_score"
		The character of metric, either "brier_score", "c_index" or "auc" depending
		on the desired metric


	Returns
	-------
	res : `np.ndarray`, shape=(n_times, )
		The matrix contains the evaluation of the desired metric
	"""
	if times is None:
		times = explainer.times
	else:
		times = np.unique(times)
	n_times = len(times)
	if isinstance(data, pd.DataFrame):
		feats = data.values
	else:
		feats = data
	surv_pred = predict(explainer, feats, times)
	survival_time =label[:, 0]
	survival_indicator = label[:, 1].astype(bool)
	res = np.zeros(n_times)
	if metric == "c_index":
		for j in range(n_times):
			res[j] = concordance_index_censored(survival_indicator, survival_time, -surv_pred[:, j])[0]

	elif metric == "brier_score":
		label_st = convert_surv_label_structarray(label)
		res = brier_score(label_st, label_st, surv_pred, times)[1]

	elif metric == "auc":
		label_st = convert_surv_label_structarray(label)
		for j in range(n_times):
			res[j] = cumulative_dynamic_auc(label_st, label_st, -surv_pred[:, j], times[j])[0]

	res = pd.DataFrame(data=np.stack([times, res]).T, columns=["times", "perf"])

	return res

def plot_performance(perf, metric, xlim=None, ylim=None):
	"""
	Plot the prediction of survival model


	Parameters
	----------

	perf :  `np.ndarray`, shape=(n_times,)
		An array of evaluation values for the desired type of metric

	metric : `str`, default = "brier_score"
		The character of metric, either "brier_score", "c_index" or "auc" depending
		on the desired metric

	xlim : `tuple`
		The x limits in size of 2.

	ylim : `tuple`
		The y limits in size of 2.

	"""
	_, ax = plt.subplots(figsize=(9, 5))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	sns.lineplot(data=perf, x="times", y="perf")
	ax.set_xlim(min(perf.times.values), max(perf.times.values))
	plt.xlabel("Times", fontsize=20)

	if xlim is not None:
		if len(xlim) != 2:
			raise ValueError("xlim should be tuple of size 2")
		else:
			xlim_left, xlim_right = xlim
	else:
		xlim_left, xlim_right = ax.get_xlim()

	if ylim is not None:
		if len(ylim) != 2:
			raise ValueError("ylim should be tuple of size 2")
		else:
			ylim_left, ylim_right = ylim
	else:
		if metric == "c_index":
			ylim_left, ylim_right = 0, 1
		elif metric == "brier_score":
			ylim_left, ylim_right = 0, .5
		elif metric == "auc":
			ylim_left, ylim_right = 0, 1
		else:
			raise ValueError("Only support output type c_index, brier_score, auc")

	ax.set_xlim(xlim_left, xlim_right)
	ax.set_ylim(ylim_left, ylim_right)

	if metric == "c_index":
		plt.ylabel("C-Index", fontsize=20)
	elif metric == "brier_score":
		plt.ylabel("Brier score", fontsize=20)
	elif metric == "auc":
		plt.ylabel("AUC", fontsize=20)
	else:
		raise ValueError("Only support output type c_index, brier_score, auc")

	plt.show()

def Brier_Score():

	return None


def C_index():

	return None


def AUC():

	return None