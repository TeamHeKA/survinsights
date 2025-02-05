import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from survinsights.prediction import predict
from survinsights.utils import order_feature_value

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def accumulated_local_effects_plots(explainer, explained_feature_name, prediction_type="survival"):
    """
    Compute accumulated local effects plots (ALE) for survival models.

    Parameters
    ----------
    explainer : Python object
        An explainer class instance for the survival model.
    explained_feature_name : str
        The feature for which ALE should be computed.
    prediction_type : str, optional, default="survival"
        Type of prediction output: "survival" or "chf". Default is "survival".

    Returns
    -------
    pd.DataFrame
        DataFrame containing ALE values for the selected feature.
    """

    if explained_feature_name in explainer.numeric_feat_names:
        return compute_numeric_ale(explainer, explained_feature_name, prediction_type)

    return compute_categorical_ale(explainer, explained_feature_name, prediction_type)


def compute_numeric_ale(explainer, expl_feat_name, prediction_type):
    """
    Compute ALE for a numeric feature by dividing it into quantile-based intervals.

    Parameters
    ----------
    explainer : Python object
        The explainer instance with data and survival labels.
    expl_feat_name : str
        The numeric feature for which ALE is computed.
    prediction_type : str
        The type of prediction output : "survival" or "chf".

    Returns
    -------
    pd.DataFrame
        DataFrame containing ALE values for the feature at different quantile levels and times.
    """
    sorted_features_df = explainer.features_df.copy(deep=True).sort_values(by=[expl_feat_name])
    grid_values = np.quantile(sorted_features_df[expl_feat_name].values, np.arange(0.0, 1.01, 0.1))
    value_group = [np.abs(grid_values[:-1] - val).argmin() for val in sorted_features_df[expl_feat_name]]

    ale_feats_df_lower, ale_feats_df_upper = create_bound_datasets(sorted_features_df, expl_feat_name, grid_values)

    survival_times = explainer.survival_labels[:, 0]
    survival_indicators = explainer.survival_labels[:, 1].astype(bool)
    unique_times = np.unique(survival_times)
    eval_times = unique_times[:-1] if unique_times[-1] >= max(survival_times[survival_indicators]) else unique_times

    if prediction_type in ["survival", "chf"]:
        lower_pred = predict(explainer, ale_feats_df_lower, eval_times, prediction_type)
        upper_pred = predict(explainer, ale_feats_df_upper, eval_times, prediction_type)
    else:
        msg = "Unsupported"
        raise ValueError(msg)

    ale_diff_df = calculate_ale_diff(lower_pred, upper_pred, value_group, eval_times)

    return finalize_ale(ale_diff_df, expl_feat_name, explainer, grid_values, eval_times)


def compute_categorical_ale(explainer, expl_feat_name, prediction_type):
    """
    Compute ALE for a categorical feature by shifting between adjacent categories.

    Parameters
    ----------
    explainer : Python object
            The explainer instance with data and survival labels.
    expl_feat_name : str
            The categorical feature for which ALE is computed.
    prediction_type : str
            Type of prediction output: "survival" or "chf"

    Returns
    -------
    pd.DataFrame
            DataFrame containing ALE values for the feature at different categories and times.
    """
    features_df = explainer.features_df.copy(deep=True)
    sorted_feat_val = order_feature_value(explainer, expl_feat_name)
    feat_name_ext = [feat_col for feat_col in features_df.columns.values if expl_feat_name in feat_col]
    value_group = [sorted_feat_val.tolist().index(val.tolist()) for val in features_df[feat_name_ext].values]

    survival_times = explainer.survival_labels[:, 0]
    survival_indicators = explainer.survival_labels[:, 1].astype(bool)
    unique_times = np.unique(survival_times)
    eval_times = unique_times[:-1] if unique_times[-1] >= max(survival_times[survival_indicators]) else unique_times

    if prediction_type in ["survival", "chf"]:
        inc_group_sel = [group > 0 for group in value_group]
        dec_group_sel = [group < len(sorted_feat_val) - 1 for group in value_group]
        feat_inc, feat_dec = (
            features_df.copy(deep=True)[inc_group_sel],
            features_df.copy(deep=True)[dec_group_sel],
        )
        feat_inc[feat_name_ext] = [sorted_feat_val[group - 1] for group in value_group if group > 0]
        feat_dec[feat_name_ext] = [
            sorted_feat_val[group + 1] for group in value_group if group < len(sorted_feat_val) - 1
        ]
        inc_pred = predict(explainer, feat_inc, eval_times, prediction_type)[["id", "pred", "times"]]
        dec_pred = predict(explainer, feat_dec, eval_times, prediction_type)[["id", "pred", "times"]]
        org_inc_pred = predict(explainer, features_df[inc_group_sel], eval_times)[["id", "pred", "times"]]
        org_dec_pred = predict(explainer, features_df[dec_group_sel], eval_times)[["id", "pred", "times"]]

        n_times = len(eval_times)
        inc_group = [group for group in value_group if group > 0]
        inc_groups_ext = np.repeat(inc_group, n_times)
        dec_group = [group + 1 for group in value_group if group < len(sorted_feat_val) - 1]
        dec_groups_ext = np.repeat(dec_group, n_times)
        inc_pred["pred"] = org_inc_pred["pred"].values - inc_pred["pred"].values
        dec_pred["pred"] = dec_pred["pred"].values - org_dec_pred["pred"].values
        inc_pred["groups"] = inc_groups_ext
        dec_pred["groups"] = dec_groups_ext
        ale_diff_df = pd.concat([inc_pred, dec_pred], ignore_index=True)
    else:
        msg = "Unsupported output type"
        raise ValueError(msg)

    return finalize_ale(ale_diff_df, expl_feat_name, explainer, sorted_feat_val, eval_times)


def create_bound_datasets(features_df, expl_feat_name, grid_values):
    """
    Create datasets with feature values shifted to lower and upper quantile boundaries.

    Parameters
    ----------
    features_df : pd.DataFrame
            Original data for ALE computation.
    expl_feat_name : str
            The feature to be shifted.
    grid_values : np.ndarray
            Quantile-based grid values for the feature.

    Returns
    -------
    tuple of pd.DataFrame
            Two DataFrames with the feature values set to the lower and upper quantile boundaries.
    """
    values_idx = [np.abs(grid_values[:-1] - val).argmin() for val in features_df[expl_feat_name]]
    ale_feats_df_lower, ale_feats_df_upper = (
        features_df.copy(deep=True),
        features_df.copy(deep=True),
    )
    ale_feats_df_lower[expl_feat_name] = np.array([grid_values[i] for i in values_idx])
    ale_feats_df_upper[expl_feat_name] = np.array([grid_values[i + 1] for i in values_idx])

    return ale_feats_df_lower, ale_feats_df_upper


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
    ALE_group_df = ale_diff_df.groupby(["groups", "times"]).mean().reset_index()[["groups", "times", "pred"]]
    n_times = len(eval_times)
    if explainer.cate_feat_names is not None:
        if feat_name in explainer.cate_feat_names:
            tmp_df = pd.DataFrame(
                data=np.array([np.zeros(n_times), eval_times, np.zeros(n_times)]).T,
                columns=["groups", "times", "pred"],
            )
            ALE_group_df = pd.concat([tmp_df, ALE_group_df])
    groups_unique = np.unique(ALE_group_df["groups"].values)
    n_groups = len(groups_unique)
    ale_df = pd.DataFrame(columns=["groups", "times", "pred"])
    for i in range(n_groups):
        group = groups_unique[i]
        res_group = (
            ALE_group_df.loc[(ALE_group_df.groups <= group)]
            .groupby(["times"])
            .sum()
            .reset_index()[["groups", "times", "pred"]]
        )
        res_group.groups = group
        ale_df = pd.concat([ale_df, res_group], ignore_index=True)

    if feat_name in explainer.numeric_feat_names:
        group_values = np.repeat(grid_values[:-1], n_times)
    else:
        encoder = explainer.encoders[feat_name]
        group_values_ = encoder.inverse_transform(grid_values).flatten()
        group_values = np.repeat(group_values_, n_times)

    id_df = ale_diff_df[["id", "groups"]].drop_duplicates()
    ale_df_ext = ale_df.join(id_df.set_index("groups"), on="groups")
    ale_df_mean = ale_df_ext.groupby(["times"]).mean().reset_index()[["times", "pred"]]
    ale_df_mean = ale_df_mean.rename(columns={"pred": "pred_mean"})
    alec_df = ale_df.join(ale_df_mean[["times", "pred_mean"]].set_index("times"), on="times")
    alec_df["alec"] = alec_df.pred.values - alec_df.pred_mean.values
    alec_df["group_values"] = group_values

    return alec_df


def plot_ale(explainer, ale_results_df, explained_feature):
    """
    Visualize the ALE results

    Parameters
    ----------
    ale_results_df : `pd.Dataframe`
        ALE result to be visualized
    explained_feature : `str`
        Name of explained feature
    """

    _, ax = plt.subplots(figsize=(9, 5))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    if explained_feature in explainer.numeric_feat_names:
        groups = np.unique(ale_results_df.groups.values)
        group_values = np.unique(ale_results_df.group_values.values)
        group_values_norm = (group_values - min(group_values)) / (max(group_values) - min(group_values))
        n_groups = len(groups)
        cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(group_values), True), cmap="BrBG")
        for i in range(n_groups):
            group = groups[i]
            sns.lineplot(
                data=ale_results_df.loc[(ale_results_df.groups == group)],
                x="times",
                y="alec",
                color=cmap.get_cmap()(group_values_norm[i]),
            )
        plt.colorbar(cmap, orientation="vertical", label=explained_feature, ax=ax, pad=0.1)
    else:
        sns.lineplot(data=ale_results_df, x="times", y="alec", hue="group_values", ax=ax)
        plt.legend(prop={"size": 12})

    plt.xlabel("Time")
    plt.ylabel("")
    plt.title("Accumulated local effects")
    plt.show()
