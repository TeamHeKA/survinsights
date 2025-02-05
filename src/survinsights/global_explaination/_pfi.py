import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from survinsights import performance

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def permutation_feature_importance(
    explainer,
    eval_times=None,
    num_perm=10,
    loss_metric="brier_score",
    output_type="ratio",
):
    """
    Compute permutation feature importance (PFI) for survival models.

    Parameters
    ----------
    explainer : Python object
        An explainer class instance for the survival model.
    eval_times : list or np.ndarray, optional
        Times at which to evaluate performance.
    num_perm : int, optional, default=10
        Number of permutations to perform.
    loss_metric : str, optional, default="brier_score"
        Performance metric for evaluation.
    output_type : str, optional, default="ratio"
        Type of PFI - "ratio" or "loss".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the PFI values for the selected features.
    """

    features_df = explainer.features_df
    survival_labels = explainer.survival_labels
    eval_times = eval_times or explainer.times

    # Compute the baseline performance of the model.
    base_performance = performance.evaluate(
        explainer, features_df, survival_labels, times=eval_times, metric=loss_metric
    )["perf"].values

    expanded_feats_name = expand_feature_names(explainer, features_df.columns.tolist())
    feat_importance_df = pd.DataFrame(columns=["feat", "times", "perf"])
    for feature_group in expanded_feats_name:
        original_name = feature_group[0] if isinstance(feature_group, list) else feature_group

        permuted_perf = get_permuted_performance(
            explainer,
            features_df,
            survival_labels,
            eval_times,
            feature_group,
            num_perm,
            loss_metric,
        )

        feat_importance_df = update_feature_importance_df(
            feat_importance_df,
            original_name,
            eval_times,
            base_performance,
            permuted_perf,
            output_type,
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
    explainer : Python object
        An explainer instance containing feature information.
    features_names : list of str
        List of feature names.

    Returns
    -------
    feats_name_ext : list of str
        List of expanded feature names with categorical features grouped.
    """

    expanded_feats_name = explainer.numeric_feat_names.copy()
    if explainer.cate_feat_names:
        expanded_feats_name += [
            [cate_name] + [name for name in features_names if cate_name in name]
            for cate_name in explainer.cate_feat_names
            if any(cate_name in f for f in features_names)
        ]
    return expanded_feats_name


def get_permuted_performance(explainer, features_df, surv_labels, eval_times, feature_group, n_perm, metric):
    """
    Compute performance after permuting feature(s).

    Parameters
    ----------
    explainer : object
            Explainer instance.
    features_df : pd.DataFrame
            Dataframe containing features.
    surv_labels : pd.DataFrame
            Survival labels for evaluation.
    eval_times : list or np.ndarray
            Evaluation times.
    feature_group : list or str
            Feature(s) to be permuted.
    n_perm : int
            Number of permutations.
    metric : str
            Metric for evaluation.

    Returns
    -------
    np.ndarray
            Array of permuted performance values.
    """
    permuted_perf = np.zeros(len(eval_times))
    for _ in range(n_perm):
        permuted_features_df = permute_feature(features_df, feature_group)
        permuted_perf += (
            performance.evaluate(
                explainer,
                permuted_features_df,
                surv_labels,
                times=eval_times,
                metric=metric,
            )["perf"].values
            / n_perm
        )
    return permuted_perf


def permute_feature(features_df, feature_names):
    """
    Permute values for specified feature(s) in the features DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
            The feature data.
    feature_names : list or str
            Feature(s) to be permuted.

    Returns
    -------
    pd.DataFrame
            DataFrame with the specified feature(s) permuted.
    """
    permuted_features_df = features_df.copy()
    if isinstance(feature_names, list):
        # categorical feature
        permuted_features_df[feature_names[1:]] = np.random.permutation(permuted_features_df[feature_names[1:]])
    else:
        # numeric feature
        permuted_features_df[feature_names] = np.random.permutation(permuted_features_df[feature_names])
    return permuted_features_df


def update_feature_importance_df(feat_importance_df, feat_name, eval_times, base_perf, permuted_perf, output_type):
    """
    Update the feature importance DataFrame with computed values.

    Parameters
    ----------
    feat_importance_df : pd.DataFrame
            DataFrame to update.
    feat_name : str
            Name of the feature.
    eval_times : list or np.ndarray
            Evaluation times.
    base_perf : np.ndarray
            Base performance values.
    permuted_perf : np.ndarray
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

    return pd.concat(
        [
            feat_importance_df,
            pd.DataFrame(importance_feat, columns=feat_importance_df.columns),
        ],
        ignore_index=True,
    )


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
    return pd.concat(
        [
            feat_importance_df,
            pd.DataFrame(full_model_feat, columns=feat_importance_df.columns),
        ],
        ignore_index=True,
    )


def plot_pfi(results, output_type, legend_loc="lower right"):
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
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    for feat_name in feature_names:
        sns.lineplot(
            data=results[results.feat == feat_name],
            x="times",
            y="perf",
            label=feat_name,
        )

    plt.xlabel("Times")
    plt.ylabel("Brier Score Ratio" if output_type == "ratio" else "Brier Score Loss")
    plt.legend(loc=legend_loc, ncol=2, prop={"size": 12})
    plt.title("Permutation Feature Importance")
    plt.savefig("Permutation_feature_importance.pdf")
    plt.show()
