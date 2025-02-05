import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def predict(explainer, data, times=None, prediction_type="survival"):
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

    n_samples = data.shape[0]

    times = explainer.times if times is None else np.unique(times)
    n_times = len(times)

    feats = data.copy(deep=True).values if isinstance(data, pd.DataFrame) else data

    if prediction_type == "survival":
        preds = explainer.sf(feats)
    elif prediction_type == "chf":
        preds = explainer.chf(feats)
    else:
        msg = "Unsupported type"
        raise ValueError(msg)

    if isinstance(data, pd.DataFrame):
        pred_df = pd.DataFrame(columns=["id", "times", "pred"])
        for i in range(n_samples):
            for j in range(n_times):
                time_j = times[j]
                if "sksurv" in explainer.model.__module__:
                    pred_df.loc[len(pred_df)] = [i, time_j, preds[i](time_j)]
                elif "pycox" in explainer.model.__module__:
                    surv_pred_i = preds[i].values
                    time_pred_i = preds.index.values
                    idx_time_ij = (np.abs(time_pred_i - time_j)).argmin()
                    pred_ij = surv_pred_i[idx_time_ij]
                    pred_df.loc[len(pred_df)] = [i, time_j, pred_ij]
                else:
                    msg = "Unsupported model"
                    raise ValueError(msg)
        return pred_df

    pred = np.zeros((n_samples, n_times))
    for i in range(n_samples):
        for j in range(n_times):
            time_j = times[j]
            if "sksurv" in explainer.model.__module__:
                pred[i, j] = preds[i](time_j)
            elif "pycox" in explainer.model.__module__:
                surv_pred_i = preds[i].values
                time_pred_i = preds.index.values
                idx_time_ij = (np.abs(time_pred_i - time_j)).argmin()
                pred[i, j] = surv_pred_i[idx_time_ij]
            else:
                msg = "Unsupported model"
                raise ValueError(msg)
    return pred


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
    [x.set_edgecolor("black") for x in ax.spines.values()]
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
        msg = "Only support output type survival, chf, risk"
        raise ValueError(msg)

    plt.show()
