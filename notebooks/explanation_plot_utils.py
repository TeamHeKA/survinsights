"""Shared visualization and explanation utilities for PBC-Seq and Simulation."""
from dataclasses import dataclass
import numpy as np

DEFAULT_COLORS = ["r", "g", "b", "c", "m", "#E69F00", "#5E3C99", "#ff9896"]

@dataclass
class Importance:
    temporal_mean: np.ndarray
    temporal_ci: np.ndarray
    global_mean: np.ndarray
    global_ci: np.ndarray
    individual_global: np.ndarray
    n_individuals: int

def load_explanations(result_dir, dataset, kind="mean", models=("CoxSig", "DDH", "DySurv")):
    if kind not in ("mean", "med"):
        raise ValueError("kind must be 'mean' or 'med'")
    return {model: np.load(result_dir / f"shap_{dataset}_{kind}_{model}.npy", allow_pickle=True)
            for model in models}

def explanation_array(results, method_idx):
    arrays = []
    for a, b, *_ in results[method_idx]:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        values = np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
        if np.unique(values).size > 2:  # Original notebook filter; NaN counts as unique
            arrays.append(values)
    if not arrays:
        raise ValueError(f"No valid explanations at method_idx={method_idx}")
    return np.stack(arrays)

def importance_summary(results, method_idx, temporal_ci_mode="legacy"):
    values = explanation_array(results, method_idx)
    temporal_mean = np.nanmean(values, axis=0)
    individual_global = np.nansum(values, axis=-1)
    global_mean = np.nansum(temporal_mean, axis=-1)
    n = len(values)
    global_ci = (1.96 * np.std(individual_global, axis=0, ddof=1) / np.sqrt(n)
                 if n > 1 else np.full_like(global_mean, np.nan))
    if temporal_ci_mode == "legacy":
        # PBC original: 1.96 * nanstd(ddof=0) / sqrt(number of individuals)
        temporal_ci = 1.96 * np.nanstd(values, axis=0) / np.sqrt(n)
    elif temporal_ci_mode == "valid":
        n_valid = np.isfinite(values).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            temporal_ci = 1.96 * np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
        temporal_ci[n_valid < 2] = np.nan
    else:
        raise ValueError("temporal_ci_mode must be 'legacy' or 'valid'")
    return Importance(temporal_mean, temporal_ci, global_mean, global_ci, individual_global, n)

def style_axis(ax, title=None, ylim=None):
    if title is not None:
        ax.set_title(title, fontsize=20, fontweight="semibold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")
    ax.tick_params(axis="both", labelsize=15)

def plot_global(ax, mean, ci=None, labels=None, colors=DEFAULT_COLORS, separator=None, fontsize=14, rotation=0):
    x = np.arange(len(mean))
    labels = list(labels if labels is not None else x)[:len(mean)]
    ax.bar(x, mean, color=[colors[i % len(colors)] for i in x])
    if ci is not None:
        ax.errorbar(x, mean, yerr=ci, fmt="none", color="black", capsize=4)
    if separator is not None:
        ax.axvline(separator, color="red", ls="--", lw=2)
    ax.set_xticks(x, labels, rotation=rotation, fontsize=fontsize)
    ax.set_xlabel("")
    ax.set_ylabel("")

def plot_temporal(ax, mean, ci=None, sampling_times=None, n_longi_feats=None,
                  labels=None, colors=DEFAULT_COLORS, horizon_idx=None):
    if sampling_times is None:
        raise ValueError("sampling_times is required")
    n_time = min(len(sampling_times), mean.shape[1],
                 horizon_idx if horizon_idx is not None else len(sampling_times))
    n_feat = min(n_longi_feats if n_longi_feats is not None else mean.shape[0], mean.shape[0])
    for j in range(n_feat):
        color = colors[j % len(colors)]
        label = labels[j] if labels is not None else f"X{j+1}"
        ax.plot(sampling_times[:n_time], mean[j, :n_time], lw=3, color=color, label=label)
        if ci is not None:
            ax.fill_between(sampling_times[:n_time], mean[j, :n_time] - ci[j, :n_time],
                            mean[j, :n_time] + ci[j, :n_time], color=color, alpha=0.2)
    ax.set_xlabel("")
    ax.set_ylabel("")


def msk_importance_summary(results, method_idx):
    """Reproduce MSK original: finite-only unique filter, per-individual global mean,
    95% global CI, and temporal ±1 SD (NOT a confidence interval).
    """
    arrays = []
    for a, b, *_ in results[method_idx]:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        values = np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
        if np.unique(values[~np.isnan(values)]).size > 2:
            arrays.append(values)
    if not arrays:
        raise ValueError(f"No valid MSK explanations at index {method_idx}")
    values = np.stack(arrays)
    temporal_mean = np.nanmean(values, axis=0)
    temporal_sd = np.nanstd(values, axis=0)
    individual_global = np.nansum(values, axis=-1)
    global_mean = np.nanmean(individual_global, axis=0)
    n = len(values)
    global_ci = (1.96 * np.nanstd(individual_global, axis=0, ddof=1) / np.sqrt(n)
                 if n > 1 else np.zeros_like(global_mean))
    return Importance(temporal_mean, temporal_sd, global_mean, global_ci,
                      individual_global, n)
