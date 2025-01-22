import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from survinsights.local_explaination import individual_conditional_expectation

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def partial_dependence_plots(
    explainer,
    explained_feature_name,
    num_samples=100,
    num_grid_points=50,
    prediction_type="survival",
):
    """
    Compute partial dependence plot (PDP)


        Parameters
        ----------
        explainer : Python object
                A Python class instance to explain the survival model.
        explained_feature_name : str
                The name of the feature to generate the PDP for.
        num_samples : int, optional, default=100
                Number of samples for calculating aggregated profiles.
        num_grid_points : int, optional, default=50
                Number of grid points for calculating aggregated profiles.
        prediction_type : str, optional, default="survival"
                Type of output to generate - options are "risk", "survival", or "chf".

    Returns
    -------
    DataFrame containing the PDP values for the selected feature.
    """
    expl_f_name = explained_feature_name
    ICE_df = individual_conditional_expectation(explainer, expl_f_name, num_samples, num_grid_points, prediction_type)

    PDP_df = ICE_df.groupby([expl_f_name, "times"]).mean().reset_index()[[expl_f_name, "times", "pred"]]
    PDP_df["prediction_type"] = prediction_type

    return PDP_df


def plot_pdp(explainer, pdp_results_df, ylim=None):
    """
    Visualize the Partial Dependence Plot (PDP) results.

    Parameters
    ----------
    explainer : Python object
            An explainer object containing feature information.
    pdp_results_df : pd.DataFrame
            A dataframe containing the PDP results to visualize.
    """

    _, ax = plt.subplots(figsize=(9, 5))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    explained_feature_name = next(col for col in pdp_results_df.columns.values if col not in ["id", "times", "pred"])
    if explained_feature_name in explainer.numeric_feat_names:
        unique_values = np.unique(pdp_results_df[explained_feature_name].values)
        normalized_values = (unique_values - min(unique_values)) / (max(unique_values) - min(unique_values))
        cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, max(unique_values), True), cmap="BrBG")
        for i, value in enumerate(unique_values):
            subset = pdp_results_df[pdp_results_df[explained_feature_name] == value]
            sns.lineplot(
                data=subset,
                x="times",
                y="pred",
                color=cmap.get_cmap()(normalized_values[i]),
                ax=ax,
            )

        plt.colorbar(cmap, ax=ax, orientation="vertical", label=explained_feature_name)
    else:
        res_sorted = pdp_results_df.sort_values(by=explained_feature_name)
        sns.lineplot(data=res_sorted, x="times", y="pred", hue=explained_feature_name, ax=ax)

    if ylim is not None:
        ylim_lower, ylim_upper = ylim
    else:
        ylim_lower, ylim_upper = 0, 1
    ax.set_ylim(ylim_lower, ylim_upper)
    plt.xlabel("Time")
    if "survival" in pdp_results_df.prediction_type.values:
        plt.ylabel("Survival function")
    elif "chf" in pdp_results_df.prediction_type.values:
        plt.ylabel("Cumulative hazard function")
    else:
        plt.ylabel("Hazard function")
    model_name = explainer.model.__class__.__name__
    if model_name == "RandomSurvivalForest":
        model_name = "RSF"
    plt.title(f"Partial Dependence Plot for the {model_name} model")
    plt.savefig(
        f"PDP_model_{model_name}_feature_{explained_feature_name}.pdf",
        bbox_inches="tight",
    )
    plt.show()
