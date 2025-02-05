from itertools import combinations


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from survinsights.local_explaination import (
    individual_conditional_expectation,
    individual_conditional_expectation_2d,
)

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def feature_interaction(explainer, explained_feature_name=None, num_samples=10, num_grid_points=10):
    """
    Compute feature interaction for a given set of features.

    Parameters
    ----------
    explainer : Python object
            Explainer class instance for the survival model.
    explained_feature_name : str or None, optional
            Feature name to compute interaction with other features. If None, computes all pairwise interactions.
    num_samples : int, optional
            Number of samples for individual conditional expectation. Default is 10.
    num_grid_points : int, optional
            Number of grid points for partial dependence calculation. Default is 10.

    Returns
    -------
    pd.DataFrame
            DataFrame with H-statistic values for each feature interaction over time.
    """
    all_feat_names = explainer.feat_names
    feature_pairs = get_feature_name_pairs(all_feat_names, explained_feature_name)

    h_stat_dfs = [
        calculate_interaction_statistic(explainer, pair, num_samples, num_grid_points) for pair in feature_pairs
    ]
    return pd.concat(h_stat_dfs, ignore_index=True)


def get_feature_name_pairs(all_feat_names, expl_feat_name):
    """
    Generate feature pairs for interaction calculation.

    Parameters
    ----------
    all_feat_names : list of str
            List of feature names.
    expl_feat_name : str or None
            Single feature name to calculate interactions with, or None to compute all feature pairs.

    Returns
    -------
    list of list
            List of feature pairs to calculate interactions for.
    """
    if expl_feat_name is None:
        return [list(pair) for pair in combinations(all_feat_names, 2)]

    return [[expl_feat_name, f_name] for f_name in all_feat_names if f_name != expl_feat_name]


def calculate_interaction_statistic(explainer, fname_pair, num_samples, num_grid_points):
    """
    Calculate the H-statistic for a pair of features over time.

    Parameters
    ----------
    explainer : Python object
            Explainer class instance for the survival model.
    fname_pair : list of str
            Pairs of feature names for interaction calculation.
    num_samples : int
            Number of samples for ICE calculation.
    num_grid_points : int
            Number of grid points for partial dependence calculation.

    Returns
    -------
    pd.DataFrame
            DataFrame with H-statistic values for the feature pair over time.
    """
    ice_2d = individual_conditional_expectation_2d(explainer, fname_pair, num_samples)
    pdp_merged = compute_partial_dependence(ice_2d, fname_pair)

    for idx, fname in enumerate(fname_pair):
        ice_single = individual_conditional_expectation(explainer, fname, num_samples, num_grid_points)
        pdp_single = ice_single.groupby(["times", fname]).mean().reset_index()[["times", fname, "pred"]]
        pdp_single = pdp_single.rename(columns={"pred": f"pred_{idx + 1}"})
        pdp_merged = pdp_merged.merge(pdp_single, on=["times", fname], how="inner")

    return compute_h_statistic(pdp_merged, fname_pair)


def compute_partial_dependence(ice_2d, fname_pair):
    """
    Compute the mean partial dependence for a pair of features over time.

    Parameters
    ----------
    ice_2d : pd.DataFrame
            ICE results for a pair of features.
    fname_pair : list of str
            Names of the feature pair.

    Returns
    -------
    pd.DataFrame
            DataFrame with mean predictions for each feature combination over time.
    """
    pdp_columns = ["times", *fname_pair]
    return ice_2d.groupby(pdp_columns).mean().reset_index()[[*pdp_columns, "pred"]]


def compute_h_statistic(pdp_df, fname_pair):
    """
    Compute the H-statistic based on partial dependence values for a feature pair.

    Parameters
    ----------
    pdp_df : pd.DataFrame
            DataFrame with partial dependence values for a feature pair over time.
    fname_pair : list of str
            Names of the feature pair.

    Returns
    -------
    pd.DataFrame
            DataFrame with H-statistic values for the feature pair over time.
    """
    pdp_df["variance"] = (pdp_df["pred"] - pdp_df["pred_1"] - pdp_df["pred_2"]) ** 2
    pdp_df["squared_corr"] = pdp_df["pred"] ** 2

    variance_sum = pdp_df.groupby("times")["variance"].sum().reset_index(name="variance_sum")
    corr_sum = pdp_df.groupby("times")["squared_corr"].sum().reset_index(name="corr_sum")

    h_stat_df = variance_sum.merge(corr_sum, on="times")
    h_stat_df["H_stat"] = h_stat_df["variance_sum"] / h_stat_df["corr_sum"]
    h_stat_df["fname_1"], h_stat_df["fname_2"] = fname_pair

    return h_stat_df[["times", "H_stat", "fname_1", "fname_2"]]


def plot_feature_interaction(h_stat_df):
    """
    Plot feature interaction H-statistics over time.

    Parameters
    ----------
    h_stat_df : pd.DataFrame
           DataFrame containing H-statistic values for each feature pair over time.
    """
    fname_pairs = h_stat_df[["fname_1", "fname_2"]].drop_duplicates().values

    _, ax = plt.subplots(figsize=(9, 5))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    for fname_pair in fname_pairs:
        pair_df = h_stat_df[(h_stat_df["fname_1"] == fname_pair[0]) & (h_stat_df["fname_2"] == fname_pair[1])]
        sns.lineplot(
            data=pair_df,
            x="times",
            y="H_stat",
            label=fname_pair[0] + "+" + fname_pair[1],
        )

    plt.xlabel("Time")
    plt.ylabel("H-statistic")
    plt.legend(prop={"size": 12})
    plt.title("Feature Interaction Over Time")
    plt.show()
