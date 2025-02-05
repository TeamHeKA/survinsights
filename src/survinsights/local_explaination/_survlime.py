import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy.optimize import minimize
from sksurv.nonparametric import nelson_aalen_estimator

from survinsights.prediction import predict

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def survlime(explainer, features_df, num_neighbors=100, sample_id=0):
    """
    Compute SurvLIME to interpret the local survival model prediction.

    Parameters
    ----------
    explainer : Python object
            Instance of the explainer class for the survival model.
    features_df : pd.DataFrame
            Data for new observations to be explained.
    num_neighbors : int, optional
            Number of neighbors to generate around the target observation. Default is 100.
    sample_id : int, optional
            Index of the target observation in data. Defaults to the first observation (index 0).

    Returns
    -------
    pd.DataFrame
            DataFrame containing SurvLIME coefficients and feature importance for each feature.
    """
    regularization_factor = 1 + 1e-4
    unique_times, baseline_chf = compute_baseline_chf(explainer.survival_labels)
    baseline_chf += regularization_factor

    explained_sample_df = features_df.iloc[[sample_id]]
    neighbors, scaled_neighbors = generate_neighbors(explainer, explained_sample_df, num_neighbors)
    model_chf = predict(explainer, neighbors, unique_times, prediction_type="chf") + regularization_factor
    distance_weights, adjustment_weights = compute_weights(scaled_neighbors, model_chf)

    args = (
        model_chf,
        baseline_chf,
        neighbors,
        distance_weights,
        adjustment_weights,
        unique_times,
    )
    initial_coefs = np.zeros(features_df.shape[1])
    optimized_coefs = minimize(
        survlime_objective,
        initial_coefs,
        args=args,
        method="BFGS",
        options={"gtol": 1e-8},
    ).x

    return format_survlime_results(features_df.columns, optimized_coefs, explained_sample_df)


def generate_neighbors(explainer, explained_sample_df, num_neighbors):
    """
    Generate neighborhood points for SurvLIME by creating random samples.

    Parameters
    ----------
    explainer : Python object
        Instance of the explainer class for the survival model.
    explained_sample_df : pd.DataFrame
        Data for the target observation.
    num_neighbors : int
        Number of neighborhood points to generate.

    Returns
    -------
    neighbor_df : pd.DataFrame
        DataFrame containing generated neighbor points.
    scaled_neighbor_df : pd.DataFrame
        DataFrame of neighbor points with scaled feature values.
    """

    np.random.seed(seed=0)
    fnames_ext = explained_sample_df.columns.values
    neighbor_df = pd.DataFrame(
        data=np.random.normal(size=(num_neighbors, explained_sample_df.shape[1])),
        columns=fnames_ext,
    )
    scale_neighbor_df = neighbor_df.copy(deep=True)

    feat_names = explainer.feat_names
    data = explainer.features_df
    scale, mean = fit_scaler(data)

    idx_ext = 0
    for idx, fname in enumerate(feat_names):
        if fname in explainer.numeric_feat_names:
            sampled_values = neighbor_df[fname].values * scale[idx] + mean[idx]
            neighbor_df[fname] = sampled_values
            scale_neighbor_df[fname] = sampled_values
            idx_ext += 1
        else:
            cate_fname_ext = [col for col in fnames_ext if fname in col]
            cate_feat_counts = data.groupby(cate_fname_ext).value_counts().reset_index(name="counts")
            sampled_values = cate_feat_counts.sample(num_neighbors, replace=True, weights="counts", random_state=1)[
                cate_fname_ext
            ].values
            neighbor_df[cate_fname_ext] = sampled_values
            scale_neighbor_df[cate_fname_ext] = sampled_values
            cond = (scale_neighbor_df[cate_fname_ext] == explained_sample_df[cate_fname_ext].values).all(axis="columns")
            if np.sum(explained_sample_df[cate_fname_ext].values) == 0:
                scale_neighbor_df.loc[~cond, cate_fname_ext] = [1] + [0] * (len(cate_fname_ext) - 1)
            else:
                scale_neighbor_df.loc[~cond, cate_fname_ext] = 0
            # reset scaler for categorical features
            scale[idx_ext : idx_ext + len(cate_fname_ext)] = 1
            mean[idx_ext : idx_ext + len(cate_fname_ext)] = 0
            idx_ext += len(cate_fname_ext)

    neighbor_df.loc[0] = explained_sample_df.values
    scale_neighbor_df.loc[0] = explained_sample_df.values
    scale_neighbor_df = (scale_neighbor_df - mean) / scale

    return neighbor_df.values, scale_neighbor_df.values


def fit_scaler(data):
    """
    Fit a StandardScaler on the data to retrieve scale and mean values.

    Parameters
    ----------
    data : pd.DataFrame
            Data used to fit the scaler.

    Returns
    -------
    scale : np.ndarray
            Scale values from the fitted scaler.
    mean : np.ndarray
            Mean values from the fitted scaler.
    """
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    scaler.fit(data)
    return scaler.scale_, scaler.mean_


def compute_baseline_chf(survival_labels):
    """
    Compute the Nelson-Aalen baseline cumulative hazard function.

    Parameters
    ----------
    survival_labels : np.ndarray
            Survival times and event indicators.

    Returns
    -------
    unique_times : np.ndarray
            Unique event times.
    baseline_chf : np.ndarray
            Baseline cumulative hazard function.
    """
    survival_times, survival_indicators = (
        survival_labels[:, 0],
        survival_labels[:, 1].astype(bool),
    )
    unique_times, baseline_chf = nelson_aalen_estimator(survival_indicators, survival_times)

    return unique_times[:-1], baseline_chf[:-1]


def compute_weights(scaled_neighbors, model_chf):
    """
    Compute weights for distances and model adjustment.

    Parameters
    ----------
    scaled_neighbors : np.ndarray
            Scaled neighborhood points.
    model_chf : np.ndarray
            Cumulative hazard function predictions.

    Returns
    -------
    distance_weights : np.ndarray
            Distance weights for the neighbors.
    adjustment_weights : np.ndarray
            Adjustment weights for model predictions.
    """
    distances = np.linalg.norm(scaled_neighbors - scaled_neighbors[0], axis=1)
    kernel_width = np.sqrt(scaled_neighbors.shape[1]) * 0.75
    distance_weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))
    adjustment_weights = model_chf / np.log(model_chf)
    return distance_weights, adjustment_weights


def survlime_objective(
    x,
    model_chf,
    baseline_chf,
    neighbors,
    distance_weights,
    adjustment_weights,
    unique_times,
):
    """
    Objective function for SurvLIME to compute feature importance.

    Parameters
    ----------
    x : np.ndarray
            Coefficients for SurvLIME.
    model_chf : np.ndarray
            Cumulative hazard function predictions for neighbors.
    baseline_chf : np.ndarray
            Baseline cumulative hazard function.
    neighbors : np.ndarray
            Neighborhood points.
    distance_weights : np.ndarray
            Weights for distances.
    adjustment_weights : np.ndarray
            Adjustment weights for model predictions.
    unique_times : np.ndarray
            Unique event times.

    Returns
    -------
    float
            Computed objective function value.
    """
    dt = unique_times[1:] - unique_times[:-1]
    log_diff = np.log(model_chf[:, :-1]) - np.log(baseline_chf[:-1]) - neighbors @ x[:, None]
    weighted_squared_error = (adjustment_weights[:, :-1] ** 2) * ((log_diff**2) * dt)
    return np.sum(distance_weights * weighted_squared_error.sum(axis=1))


def format_survlime_results(features, coefficients, explained_sample):
    """
    Format SurvLIME results into a DataFrame.

    Parameters
    ----------
    features : pd.Index
            Feature names.
    coefficients : np.ndarray
            Coefficients from SurvLIME optimization.
    explained_sample : pd.DataFrame
            Data of the explained observation.

    Returns
    -------
    pd.DataFrame
            DataFrame containing features, coefficients, and feature importance.
    """
    feature_importance = coefficients * explained_sample.values.flatten()
    return pd.DataFrame(
        {
            "Feature": features,
            "Coefficient": coefficients,
            "Feature Importance": feature_importance,
        }
    )


def plot_survlime(results_df, sample_id=0):
    """
    Plot the SurvLIME feature importance results.

    Parameters
    ----------
    results_df : pd.DataFrame
            DataFrame containing SurvLIME results with feature coefficients and importance.
    sample_id : int, optional
            ID of the observation being visualized. Default is 0.
    """

    _, ax = plt.subplots(figsize=(9, 5))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    colors = ["g" if c >= 0 else "r" for c in results_df["Coefficient"].values]
    sns.barplot(data=results_df, y="Feature", x="Feature Importance", palette=colors)

    plt.xlabel("Feature Importance")
    plt.ylabel("")
    plt.title(f"SurvLIME Feature Importance for Observation ID = {sample_id}")
    plt.show()
