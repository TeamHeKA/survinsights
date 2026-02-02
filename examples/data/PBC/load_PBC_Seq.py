import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

def load():
    # load PBC Seq
    robjects.r.source(os.getcwd() + "/examples/data/PBC/load_PBC_Seq.R")
    static_feat_list = ['edema', 'age']
    longi_feat_list = ['serBilir', 'albumin', 'alkaline', 'SGOT', 'prothrombin']
    data_R = robjects.r["load"]()

    with localconverter(default_converter + pandas2ri.converter):
        df_org = robjects.conversion.rpy2py(data_R)
    df_org = df_org[(df_org[longi_feat_list] > -1e-4).all(axis=1)]
    print(df_org.columns)

    # TODO: CHECK CHECK
    for feat in longi_feat_list:
        df_org[feat] = np.log(df_org[feat].values)

    # creating instance of one-hot-encoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # perform one-hot encoding on time-indep columns
    df_org['edema'] = encoder.fit_transform(df_org[['edema']]).toarray()
    df_org['age'] = (df_org['age'].values - df_org['age'].values.min()) / (df_org['age'].values.max() - df_org['age'].values.min())
    df_org['id'] = df_org['id'].astype(int)

    # Extraction
    df_org['id_count'] = df_org.groupby('id')['id'].transform('count')
    df_org = df_org[df_org['id_count'] > 1]
    df_org["times"] = df_org["times"].values.round(2)
    df_org["tte"] = df_org["tte"].values.round(2)
    surv_times, surv_inds = tuple(df_org[["id", "tte", "label"]].drop_duplicates("id")[["tte", "label"]].values.T)
    surv_times = surv_times.round(2)

    step = .05
    end_time = np.max(df_org[["times", "tte"]].values)
    sampling_times = np.arange(0, stop=end_time, step=step)
    n_sampling_times = len(sampling_times)
    longi_markers = ['serBilir', 'albumin', 'alkaline', 'SGOT', 'prothrombin']
    # longi_markers = ['serBilir', 'albumin', 'prothrombin']
    n_longi_markers = len(longi_markers)
    idxs = np.unique(df_org.id.values)
    n_samples = len(idxs)

    df_ = df_org[["id", "tte", "label", "times"] + longi_markers]
    X = np.zeros((n_samples, n_sampling_times, 1 + n_longi_markers), dtype=np.single)
    for i in np.arange(n_samples):
        idx = idxs[i]
        df_times = pd.DataFrame(np.array((np.array([idx] * n_sampling_times), sampling_times)).T, columns=["id", "times"])
        df_idx = pd.merge(df_times, df_[df_.id == idx], how="left", on=["id", "times"]).fillna(method='ffill')
        X[i] = df_idx[["times"] + longi_markers].values

    paths = torch.from_numpy(X.copy())
    surv_labels = np.array([surv_times, surv_inds], dtype=np.single).T
    bin_df = df_org[["id"] + static_feat_list].drop_duplicates("id").sort_values(by=['id'])
    static_feat = bin_df[static_feat_list].values
    time_scale = 1 / step
    ddh_info_sup = (longi_markers, static_feat_list, time_scale, bin_df)

    truncated_paths = truncate_paths_at_events(paths, sampling_times, surv_labels)
    visualize(paths, truncated_paths, surv_labels, longi_feat_list, save_path="figures/PBC_data.pdf")

    return truncated_paths, surv_labels, longi_feat_list, static_feat, static_feat_list, ddh_info_sup

def truncate_paths_at_events(
    paths: torch.Tensor,
    sampling_times: np.ndarray,
    surv_labels: np.ndarray,
) -> torch.Tensor:
    """
    Freeze each subject's longitudinal channels at their event time.
    """
    truncated = paths.clone()
    n_subjects, _, _ = paths.shape
    n_times = sampling_times.shape[0]

    event_idx = np.where(surv_labels[:, 1] == 1)[0]  # uncensored only
    for idx in event_idx:
        t_event = surv_labels[idx, 0]
        tpos = int(np.searchsorted(sampling_times, t_event))
        if 0 <= tpos < n_times - 1:
            truncated[idx, tpos:, 1:] = truncated[idx, tpos, 1:]
    return truncated

def visualize(
    paths: torch.Tensor,
    truncated_paths: torch.Tensor,
    surv_labels: np.ndarray,
    feat_list: list,
    sel_idx: int = 1,
    save_path: str | None = "simu_data.pdf",
    show: bool = True,
) -> None:
    """
    Visualize subject paths, feature importance, and survival histogram.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_edgecolor("black")

    colors = ["r", "g", "b", "c", "m"]
    times = paths[sel_idx, :, 0]
    dt = times[1] - times[0]

    # ---- longitudinal paths
    for l, ts in enumerate(truncated_paths[sel_idx, :, 1:].T):
        axs[0].plot(times, ts, color=colors[l], label=feat_list[l])
    sampling_times = paths[sel_idx,:,0]
    time_step = sampling_times[1] - sampling_times[0]
    axs[0].set_title("Longitudinal features", fontweight="semibold", fontsize=15)
    axs[0].legend(ncols=2, fontsize=10, loc="upper left")
    axs[0].set_xlabel("") 
    axs[0].tick_params(axis='x', labelsize=12) 
    axs[0].tick_params(axis='y', labelsize=12) 
    axs[0].set_ylabel("") 
    axs[0].set_xlim(min(sampling_times), max(sampling_times) + 10 * time_step)

    # ---- survival histogram
    surv_times, surv_inds = surv_labels[:, 0], surv_labels[:, 1]
    axs[1].hist(surv_times, bins=100, density=True)
    cens_pct = (1 - surv_inds.sum() / len(surv_inds)) * 100
    axs[1].legend([Line2D([0], [0], lw=2)], [f"Censoring: {cens_pct:.1f}%"], loc="upper right")
    axs[1].set_title("Histogram of survival time", fontweight="semibold", fontsize=15)
    axs[1].set_xlabel("") 
    axs[1].tick_params(axis='x', labelsize=12) 
    axs[1].tick_params(axis='y', labelsize=12) 
    axs[1].set_ylabel("") 
    axs[1].set_xlim(min(sampling_times), max(sampling_times) + 10 * time_step)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)