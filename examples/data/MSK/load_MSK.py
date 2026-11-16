import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import os
# os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

def load():
    # n_sel_samples = 2000
    paths = np.load(os.getcwd() + "/../examples/data/MSK/paths.npy")
    surv_labels = np.load(os.getcwd() + "/../examples/data/MSK/surv_labels.npy")
    static_feat = np.load(os.getcwd() + "/../examples/data/MSK/static_feat.npy")
    
    paths = torch.from_numpy(paths.copy()).clone()
    sampling_times = paths[0, :, 0]
    longi_feat_list = ["Cancer antigen 15-3", "Carcinoembryonic antigen"]
    # static_feat_list = ['Sex', 'HER2', 'Current Age']
    static_feat_list = ['Stage', 'Current Age', 'Smoking', 'HR', 'HER2']

    truncated_paths = truncate_paths_at_events(paths, sampling_times, surv_labels)
    visualize(paths, truncated_paths, surv_labels, save_path="../figures/MSK_data.pdf")
    time_scale = (1 / (sampling_times[1] - sampling_times[0])).item()
    bin_df = pd.DataFrame(data=static_feat, columns=static_feat_list)
    bin_df["id"] = np.arange(paths.shape[0])
    ddh_info_sup = (longi_feat_list, static_feat_list, time_scale, bin_df)

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

    # ---- longitudinal paths
    for l, ts in enumerate(truncated_paths[sel_idx, :, 1:].T):
        axs[0].plot(times, ts, color=colors[l], label=f"$X_{{{l+1}}}(t)$")
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