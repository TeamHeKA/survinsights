import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from survinsights.prediction import predict

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def individual_conditional_expectation(
    explainer,
    explained_feature_name,
    num_samples=100,
    num_grid_points=50,
    prediction_type="survival",
):
    """
    Compute Individual Conditional Expectation (ICE) for a single feature.

    Parameters
    ----------
    explainer : Python object
        Explainer instance for the survival model, containing model and data attributes.
    explained_feature_name : str
        Name of the feature for which ICE is calculated.
    num_samples : int, optional
        Number of samples to use for ICE computation. Default is 100.
    num_grid_points : int, optional
        Number of grid points to sample for the feature. Default is 50.
    prediction_type : str, optional
        Type of prediction output: "survival" or "chf". Default is "survival".

    Returns
    -------
    pd.DataFrame
        DataFrame containing ICE values for the specified feature.
    """
    features_df = explainer.features_df
    ice_features_df, expl_feat_space, expl_feat_name_ext = prepare_ice_data(
        features_df, explained_feature_name, explainer, num_samples, num_grid_points
    )

    predictions = predict(explainer, ice_features_df, prediction_type=prediction_type)

    ice_results_df = construct_ice_result_dataframe(predictions, expl_feat_space, num_samples, expl_feat_name_ext)

    if explained_feature_name not in explainer.numeric_feat_names:
        encoder = explainer.encoders[explained_feature_name]
        encoded_expl_f_name = encoder.get_feature_names_out([explained_feature_name]).tolist()
        encoded_feat_value = ice_results_df[encoded_expl_f_name].values
        ice_results_df[explained_feature_name] = encoder.inverse_transform(encoded_feat_value).flatten()
        ice_results_df = ice_results_df.drop(columns=encoded_expl_f_name)

    return ice_results_df


def individual_conditional_expectation_2d(
    explainer, explained_feature_names, num_samples=100, prediction_type="survival"
):
    """
    Compute Individual Conditional Expectation (ICE) for two features.

    Parameters
    ----------
    explainer : Python object
        Explainer instance for the survival model, containing model and data attributes.
    explained_feature_names : list of str
        List of two feature names for which ICE is computed.
    num_samples : int, optional
        Number of samples to use for ICE computation. Default is 100.
    prediction_type : str, optional
        Type of prediction output: "survival" or "chf". Default is "survival".

    Returns
    -------
    pd.DataFrame
        DataFrame containing ICE values for the specified feature pair.
    """
    features_df = explainer.features_df
    ice_features_df, expl_feats_space, expt_feats_name_ext = prepare_2d_ice_data(
        features_df, explained_feature_names, explainer, num_samples
    )

    predictions = predict(explainer, ice_features_df, prediction_type=prediction_type)

    ice_results_df = construct_ice_result_dataframe(predictions, expl_feats_space, num_samples, expt_feats_name_ext)
    for expl_f_name in explained_feature_names:
        if expl_f_name not in explainer.numeric_feat_names:
            encoder = explainer.encoders[expl_f_name]
            encoded_expl_f_name = encoder.get_feature_names_out([expl_f_name]).tolist()
            encoded_feat_value = ice_results_df[encoded_expl_f_name].values
            ice_results_df[expl_f_name] = encoder.inverse_transform(encoded_feat_value).flatten()

    return ice_results_df


# Utility functions for ICE computations
def prepare_ice_data(features_df, explained_feature_name, explainer, num_samples, num_grid_points):
    """
    Prepare data for ICE computation for a single feature.

    Parameters
    ----------
    features_df : pd.DataFrame
            DataFrame of feature values.
    explained_feature_name : str
            Name of the feature for which ICE is computed.
    explainer : Python object
            Explainer instance containing feature information.
    num_samples : int
            Number of samples to use for ICE computation.
    num_grid_points : int
            Number of grid points to sample for the feature.

    Returns
    -------
    tuple
            Extended data for ICE computation, unique feature values, and extended feature name.
    """
    if explained_feature_name in explainer.numeric_feat_names:
        expl_feat_idx = features_df.columns.get_loc(explained_feature_name)
        sub_feats_val = features_df[:num_samples].values
        expl_feat_val = sub_feats_val[:, expl_feat_idx]
        expl_feat_space = np.linspace(min(expl_feat_val), max(expl_feat_val), num_grid_points)
        feats_val_ext = np.repeat(sub_feats_val, len(expl_feat_space), axis=0)
        feats_val_ext[:, expl_feat_idx] = np.tile(expl_feat_space, num_samples)
        expl_feat_name_ext = [explained_feature_name]
    else:
        expl_feat_name_ext = [col for col in features_df.columns if explained_feature_name in col]
        expl_feat_idx = [features_df.columns.get_loc(col) for col in expl_feat_name_ext]
        sub_feats_val = features_df[:num_samples].values
        expl_feat_space = features_df[expl_feat_name_ext].drop_duplicates().values
        feats_val_ext = np.repeat(sub_feats_val, len(expl_feat_space), axis=0)
        feats_val_ext[:, expl_feat_idx] = np.tile(expl_feat_space, (num_samples, 1))

    features_df_ext = pd.DataFrame(feats_val_ext, columns=features_df.columns)

    return features_df_ext, expl_feat_space, expl_feat_name_ext


def prepare_2d_ice_data(features_df, explained_feature_names, explainer, num_samples):
    """
    Prepare data for ICE computation for a pair of features.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame of feature values.
    explained_feature_names : list of str
        List of two feature names for which ICE is computed.
    explainer : Python object
        Explainer instance containing feature information.
    num_samples : int
        Number of samples to use for ICE computation.

    Returns
    -------
    tuple
        Extended data for ICE computation, unique feature values for the feature pair, and extended feature names.
    """
    expl_feat_names_ext = []
    expl_feature_indices = []
    for expl_f_name in explained_feature_names:
        if expl_f_name in explainer.numeric_feat_names:
            expl_feat_names_ext.append(expl_f_name)
            expl_feature_indices.append(features_df.columns.get_loc(expl_f_name))
        else:
            cate_feat_name_ext = [col for col in features_df.columns if expl_f_name in col]
            expl_feat_names_ext.extend(cate_feat_name_ext)
            expl_feature_indices.extend([features_df.columns.get_loc(col) for col in cate_feat_name_ext])

    expl_feats_space = features_df[expl_feat_names_ext].drop_duplicates().values
    sub_feats_val = features_df[:num_samples].values
    feats_val_ext = np.repeat(sub_feats_val, len(expl_feats_space), axis=0)
    feats_val_ext[:, expl_feature_indices] = np.vstack([expl_feats_space] * num_samples)

    features_df_ext = pd.DataFrame(feats_val_ext, columns=features_df.columns)

    return features_df_ext, expl_feats_space, expl_feat_names_ext


def construct_ice_result_dataframe(predictions, explained_feature_space, num_samples, explained_feature_name_ext):
    """
    Construct the ICE DataFrame from predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
            Predicted values for extended data.
    explained_feature_space : np.ndarray
            Array of unique values for the selected feature(s).
    num_samples : int
            Number of samples used for ICE computation.
    explained_feature_name_ext : list of str
            Extended names of the explained feature(s) for ICE computation.

    Returns
    -------
    pd.DataFrame
            DataFrame containing ICE values for the selected feature(s).
    """
    ice_df = pd.DataFrame(columns=["id", "times", "pred", *explained_feature_name_ext])
    for i in range(num_samples):
        for j, value in enumerate(explained_feature_space):
            prediction_subset = predictions[predictions.id == float(i * len(explained_feature_space) + j)]
            for k in range(prediction_subset.shape[0]):
                ice_df.loc[len(ice_df)] = [
                    i,
                    prediction_subset.times.values[k],
                    prediction_subset.pred.values[k],
                ] + (value.tolist() if isinstance(value, np.ndarray) else [value])
    return ice_df


def plot_ice(explainer, ice_results_df, sample_id=0, xvar="Time", ylim=None):
    """
    Visualize the ICE results.

    Parameters
    ----------
    explainer : Python object
            Explainer instance for the survival model.
    ice_results_df : pd.DataFrame
            DataFrame containing ICE results to visualize.
    sample_id : int, optional
            ID of the observation to plot. Default is 0.
    xvar : str, optional
            Name of the x-axis variable. Default is "Time".
    ylim : tuple, optional
            Lower and upper limits for the y-axis. Default is None.
    """
    _, ax = plt.subplots(figsize=(9, 5))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    explained_feature_name = next(col for col in ice_results_df.columns.values if col not in ["id", "times", "pred"])
    if xvar == "Time":
        if explained_feature_name in explainer.numeric_feat_names:
            unique_values = np.unique(ice_results_df[explained_feature_name].values)
            normalized_values = (unique_values - min(unique_values)) / (max(unique_values) - min(unique_values))
            cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(unique_values)), cmap="BrBG")

            for i, value in enumerate(unique_values):
                subset = ice_results_df[
                    (ice_results_df.id == sample_id) & (ice_results_df[explained_feature_name] == value)
                ]
                sns.lineplot(
                    data=subset,
                    x="times",
                    y="pred",
                    color=cmap.get_cmap()(normalized_values[i]),
                    ax=ax,
                )

            plt.colorbar(cmap, ax=ax, orientation="vertical", label=explained_feature_name)
        else:
            subset = ice_results_df[ice_results_df.id == sample_id].sort_values(by=explained_feature_name)
            sns.lineplot(data=subset, x="times", y="pred", hue=explained_feature_name, ax=ax)
    else:
        unique_times = np.unique(ice_results_df["times"].values)
        normalized_times = (unique_times - min(unique_times)) / (max(unique_times) - min(unique_times))
        cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(unique_times)), cmap="BrBG")

        for i, time in enumerate(unique_times):
            subset = ice_results_df[(ice_results_df.id == sample_id) & (ice_results_df["times"] == time)]
            sns.lineplot(
                data=subset,
                x=explained_feature_name,
                y="pred",
                color=cmap.get_cmap()(normalized_times[i]),
                ax=ax,
            )

        plt.colorbar(cmap, ax=ax, orientation="vertical", label="Times")

    if ylim is not None:
        ylim_lower, ylim_upper = ylim
    else:
        ylim_lower, ylim_upper = 0, 1
    ax.set_ylim(ylim_lower, ylim_upper)
    if xvar == "Time":
        plt.xlabel("Time")
    else:
        plt.xlabel(explained_feature_name)
    # plt.xlabel("Time")
    plt.ylabel("Survival prediction")
    plt.title(f"ICE for feature {explained_feature_name} of observation id = {sample_id}")
    plt.savefig(
        f"ICE_feature_{explained_feature_name}_of_id={sample_id}.pdf",
        bbox_inches="tight",
    )
    plt.show()
