import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')
from examples.learner.dynamic_deephit_ext import construct_df, Dynamic_DeepHit_ext

def weighted_median(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)

    # sort by values
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    # cumulative sum of weights
    cum_weights = np.cumsum(weights)

    # weighted median
    return values[cum_weights >= 0.5][0]

def coxsig_predict(explainer, longi_feat, pred_times=None, prediction_type="survival", static_feat=None):
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

    prediction_type : `str`, default = "survival"
        The character of output type, either "risk", "survival" or "chf" depending
        on the desired output


    Returns
    -------
    pred : `np.ndarray`, shape=(n_samples, n_times)
        The matrix contains the prediction
	"""

	if prediction_type == "survival":
		surv_preds = explainer.sf(longi_feat, pred_times, static_feat=static_feat)
		preds = []
		eps = 1e-4
		time_interval = explainer.model.sampling_times[1] - explainer.model.sampling_times[0]
		for j in np.arange(len(pred_times)):
			pred_at_t = surv_preds[j].sum(axis=1) * time_interval
			preds.append(pred_at_t)

	if prediction_type == "survival_ind":
		surv_preds = explainer.sf(longi_feat, pred_times, static_feat=static_feat)
		preds = []
		for j in np.arange(len(pred_times)):
			preds.append(surv_preds[j])
			# preds.append(surv_preds[j][:, ])

		return np.array(preds)


	if prediction_type == "med-survival":
		surv_preds = explainer.sf(longi_feat, pred_times, static_feat=static_feat)
		preds = []
		eps = 1e-4
		time_interval = explainer.model.sampling_times[1] - explainer.model.sampling_times[0]
		for j in np.arange(len(pred_times)):

			p = surv_preds[j][:, :-1] - surv_preds[j][:, 1:]   # shape (n-1,)
			# optional: ensure numerical safety
			p = np.clip(p, 0, None)
			p = (p.T / p.sum(axis=1)).T

			pred_at_t = []
			for i in range(p.shape[0]):
				pred_at_t.append(time_interval * weighted_median(np.arange(p.shape[1]), p[i]))
			preds.append(np.array(pred_at_t))

	
	if prediction_type == "risk":
		risk_preds = explainer.risk(longi_feat, pred_times, static_feat=static_feat)
		preds = []
		eps = 1e-4
		for j in np.arange(len(pred_times)):
			pred_time = pred_times[j]
			t_pred_id = np.searchsorted(explainer.model.sampling_times[1:], pred_time + eps)
			n_sampling_times = len(explainer.model.sampling_times)
			np.random.seed(0)
			sel_idx = np.random.choice(np.arange(t_pred_id, n_sampling_times))
			pred_at_t = risk_preds[:, j, sel_idx]
			preds.append(pred_at_t)

	return np.array(preds).flatten()

def ddh_predict(explainer, longi_feat, pred_times, prediction_type="survival", static_feat=None):
	"""
	Calculate model prediction in a unified way (Dynamic DeepHit)


	Parameters
    ----------
    explainer :  `class`
        A Python class used to explain the survival model


    Returns
    -------
    pred : `np.ndarray`, shape=(n_samples, n_times)
        The matrix contains the prediction
	"""
	n_samples = longi_feat.shape[0]
	cont_feat, bin_feat, time_scale, _ = explainer.model.ddh_info_sup
	sampling_times = longi_feat[0, :, 0]
	ps_bin_df = pd.DataFrame(data=static_feat, columns=bin_feat)
	ps_bin_df["id"] = np.arange(n_samples)
	ps_surv_labels = np.hstack((np.max(1.1 * np.array(sampling_times)) * np.ones(n_samples).reshape(-1, 1),
							 np.random.randint(0, 1, n_samples).reshape(-1, 1)))
	df = construct_df(longi_feat.clone(), ps_surv_labels, cont_feat, bin_feat, time_scale, ps_bin_df)
	ddh_model_tmp = Dynamic_DeepHit_ext()
	(data, _, _), _, (data_mi) = ddh_model_tmp.preprocess(df, cont_feat, bin_feat)

	if prediction_type == "survival":
		surv_preds = explainer.sf(data, data_mi, pred_times * time_scale)
		preds = []
		for j in np.arange(len(pred_times)):
			pred_at_t = (surv_preds[j] * np.arange(1, surv_preds[j].shape[1] + 1)).sum(axis=1)
			preds.append(pred_at_t)
	
	if prediction_type == "survival_ind":
		surv_preds = explainer.sf(data, data_mi, pred_times * time_scale)
		preds = []
		for j in np.arange(len(pred_times)):
			pred_at_t = surv_preds[j][:, 1]
			preds.append(pred_at_t)

	if prediction_type == "med-survival":
		surv_preds = explainer.sf(data, data_mi, pred_times * time_scale)
		preds = []
		# for j in np.arange(len(pred_times)):
		# 	p = surv_preds[j]
		# 	pred_at_t = []
		# 	for i in range(p.shape[0]):
		# 		pred_at_t.append(weighted_median(np.arange(p.shape[1]), p[i]))
		# 	preds.append(np.array(pred_at_t))
		for j in range(len(pred_times)):
			surv = np.asarray(surv_preds[j])
			crossed = surv <= 0.5
			median_idx = np.where(crossed.any(axis=1), crossed.argmax(axis=1), np.inf)
			preds.append(median_idx / time_scale)
	return np.array(preds).flatten()


# ---------- Unified DySurv prediction ----------

# ============================================================
# Unified DySurv prediction
# ============================================================

def dysurv_predict(
    explainer, longi_feat, pred_times,
    prediction_type="survival", static_feat=None,
    horizon=None,
):
    pred_times = np.atleast_1d(pred_times).astype(float)
    survival = explainer.sf(
        longi_feat, pred_times, static_feat=static_feat
    )

    times = np.asarray(explainer.model.sampling_times, dtype=float)

    if prediction_type == "survival_ind":
        return survival

    preds = []

    for j, t_p in enumerate(pred_times):
        mask = times > t_p

        grid = np.r_[t_p, times[mask]]
        curves = np.column_stack([
            np.ones(survival.shape[1]),
            survival[j][:, mask],
        ])

        if prediction_type == "survival":
            # Restricted mean residual survival time
            pred = np.trapz(curves, grid, axis=1)

        elif prediction_type == "med-survival":
            # Median residual survival time
            pred = np.full(curves.shape[0], np.inf)

            for i, s in enumerate(curves):
                crossed = np.flatnonzero(s <= 0.5)

                if len(crossed) == 0:
                    continue

                k = crossed[0]

                if k == 0:
                    pred[i] = 0.0
                else:
                    t0, t1 = grid[k - 1:k + 1]
                    s0, s1 = s[k - 1:k + 1]

                    if s0 == s1:
                        t_median = t1
                    else:
                        t_median = t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0)

                    pred[i] = t_median - t_p

        elif prediction_type == "risk":
            if horizon is None:
                target = times[-1]
            else:
                target = t_p + horizon

            if target < t_p or target > times[-1]:
                raise ValueError("Risk horizon must lie within the prediction grid.")

            pred = 1.0 - np.array([
                np.interp(target, grid, s)
                for s in curves
            ])

        else:
            raise ValueError(
                f"Unsupported prediction type: {prediction_type}"
            )

        preds.append(pred)

    return np.asarray(preds).flatten()


def plot_prediction(pred, prediction_type):
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
	if prediction_type == "survival":
		ax.set_ylim(0, 1)
		plt.ylabel("Survival function", fontsize=20)
	elif prediction_type == "chf":
		plt.ylabel("Cumulative hazard function", fontsize=20)
	elif prediction_type == "risk":
		plt.ylabel("Hazard function", fontsize=20)
	else:
		raise ValueError("Only support output type survival, chf, risk")

	plt.show()