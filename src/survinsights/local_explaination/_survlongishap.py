"""DynSurvX: two-phase feature and temporal importance estimation."""

# ============================================================
# Imports
# ============================================================

import math
from itertools import product
from typing import Sequence, Tuple

import numpy as np
import shap
import torch

from src.survinsights.longi_prediction import (
    coxsig_predict,
    ddh_predict,
    dysurv_predict,
)


# ============================================================
# Prediction and weighted regression utilities
# ============================================================

def _predict_for_explanation(
    explainer, longi_feat, pred_times,
    prediction_type="survival", static_feat=None,
):
    predictors = {
        "coxsig": coxsig_predict,
        "ddh": ddh_predict,
        "dysurv": dysurv_predict,
    }
    model_name = explainer.model_name.lower()

    if model_name not in predictors:
        raise ValueError(f"Unsupported model: {model_name}")

    return predictors[model_name](
        explainer, longi_feat, pred_times,
        prediction_type=prediction_type, static_feat=static_feat,
    )


def _shap_kernel_weight(s: int, m: int) -> float:
    if s == 0 or s == m:
        return 1.0  # Preserve the endpoint convention in the original script.
    return (m - 1) / (math.comb(m, s) * s * (m - s))


def _soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def weighted_lasso_coordinate_descent(
    A, b, lam=10.0, sample_weight=None,
    max_iter=1000, tol=1e-6, intercept=True,
):
    """Weighted L1-penalized least squares via coordinate descent.

    Preserves the original solver's convention: when intercept=True,
    the appended intercept is also penalized, then omitted on return.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if intercept:
        A = np.column_stack((A, np.ones(A.shape[0])))

    n, d = A.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    if w.shape != (n,):
        raise ValueError("sample_weight must have length n_samples")

    Q = A.T @ (w[:, None] * A) + 1e-6 * np.eye(d)
    q = A.T @ (w * b)
    coef = np.zeros(d, dtype=float)

    for _ in range(max_iter):
        previous = coef.copy()

        for j in range(d):
            rho = q[j] - Q[j] @ coef + Q[j, j] * coef[j]
            coef[j] = _soft_threshold(rho, lam) / Q[j, j] if Q[j, j] > 0 else 0.0

        if np.linalg.norm(coef - previous) < tol:
            break

    return coef[:-1] if intercept else coef


def _weighted_least_squares_lasso(coals, weights, predictions, reg, intercept):
    return weighted_lasso_coordinate_descent(
        coals, predictions, lam=reg,
        sample_weight=weights, intercept=intercept,
    )


# ============================================================
# Coalition construction and temporal splitting
# ============================================================

def _build_coalition_tensor_from_data(
    base: Tuple[torch.Tensor, torch.Tensor],
    time_col: int,
    longi_feat_col_indices: Sequence[int],
    coals: Tuple[np.ndarray, np.ndarray],
    replacement: Tuple[np.ndarray, np.ndarray],
):
    """Build perturbed longitudinal/static data for feature coalitions.

    Retains the original perturbation rule: absent longitudinal features
    are set to zero; present features are copied from the explained subject.
    """
    expl_longi, expl_static = base
    longi_coals, static_coals = coals
    longi_replacement, static_replacement = replacement

    K, _ = longi_coals.shape
    expl_longi = torch.as_tensor(expl_longi, dtype=torch.float32)
    longi_replacement = torch.as_tensor(longi_replacement, dtype=expl_longi.dtype)

    idx = torch.randint(longi_replacement.shape[0], (1,)).item()
    longi_data = longi_replacement[idx:idx + 1].repeat(K, 1, 1)
    longi_data[:, :, time_col] = expl_longi[:, time_col]

    for j, col in enumerate(longi_feat_col_indices):
        keep = torch.as_tensor(longi_coals[:, j] == 1)
        longi_data[keep, :, col] = expl_longi[:, col]
        longi_data[~keep, :, col] = 0.0

    static_data = None

    if static_coals is not None:
        idx = np.random.randint(static_replacement.shape[0])
        static_data = np.repeat(np.asarray(static_replacement[idx:idx + 1]), K, axis=0)
        keep = static_coals.astype(bool)
        static_data[keep] = np.broadcast_to(np.asarray(expl_static), static_data.shape)[keep]

    return longi_data, static_data


def _split_intervals(
    intervals: Sequence[Tuple[int, int]], split_mask: np.ndarray,
) -> list[Tuple[int, int]]:
    """Bisect selected half-open intervals [start, end)."""
    new_intervals = []

    for (start, end), do_split in zip(intervals, split_mask.astype(bool)):
        if not do_split or end - start <= 1:
            new_intervals.append((start, end))
        else:
            mid = (start + end) // 2
            new_intervals.extend([(start, mid), (mid, end)])

    return new_intervals


# ============================================================
# Phase 1: overall feature importance and screening
# ============================================================

def _estimate_feature_importance(
    explainer, expl_longi, expl_static, pred_t, prediction_type,
    n_longi_feats, n_static_feats, replacement_mask, reg, intercept, rng,
):
    n_features = n_longi_feats + n_static_feats
    coalitions = np.array(list(product([0, 1], repeat=n_features)), dtype=int)
    coalitions = coalitions[rng.permutation(len(coalitions))]
    weights = np.array([
        _shap_kernel_weight(int(coal.sum()), n_features)
        for coal in coalitions
    ])

    static_background = (
        np.asarray(explainer.static_feats)[replacement_mask]
        if n_static_feats else None
    )
    longi_data, static_data = _build_coalition_tensor_from_data(
        base=(expl_longi, expl_static),
        time_col=0,
        longi_feat_col_indices=list(range(1, n_longi_feats + 1)),
        coals=(coalitions[:, :n_longi_feats],
               coalitions[:, n_longi_feats:] if n_static_feats else None),
        replacement=(explainer.paths[replacement_mask], static_background),
    )

    importance = np.zeros(n_features)
    t_eval = np.array([pred_t], dtype=float)

    for k in range(3):
        predictions = np.asarray(_predict_for_explanation(
            explainer, longi_data, t_eval,
            prediction_type=prediction_type, static_feat=static_data,
        )).reshape(-1)

        coefficients = _weighted_least_squares_lasso(
            coalitions, weights, predictions, reg * 0.75**k, intercept,
        )
        importance = np.abs(coefficients)

        if importance.sum() > 1e-12:
            break

    total = importance.sum()
    return importance / total if total > 1e-12 else np.full(n_features, 1 / n_features)


def _should_refine(feature_importance, delta_1):
    n_features = len(feature_importance)
    entropy = -np.sum(feature_importance * np.log(feature_importance + 1e-12))
    return (
        entropy < np.log(n_features) - 0.1
        or np.any(feature_importance > delta_1)
    )


# ============================================================
# Phase 2: interval-wise importance and adaptive refinement
# ============================================================

def _estimate_interval_importance(
    explainer, sel_idx, expl_longi, expl_static, col, intervals,
    bg_idx, pred_t, prediction_type, longi_dtype,
):
    n_static_feats = 0 if expl_static is None else len(expl_static)
    n_intervals = len(intervals)
    t_eval = np.array([pred_t], dtype=float)

    def wrapper_predict(x):
        x = np.asarray(x)
        n_batch = x.shape[0]
        longi_batch = torch.as_tensor(
            np.repeat(expl_longi[None, :, :], n_batch, axis=0),
            dtype=longi_dtype,
        )
        static_batch = (
            np.repeat(np.asarray(expl_static)[None, :], n_batch, axis=0)
            if n_static_feats else None
        )

        for i, row in enumerate(x):
            for j, (start, end) in enumerate(intervals):
                path_idx = int(round(float(row[j])))
                longi_batch[i, start:end, col] = torch.as_tensor(
                    explainer.paths[path_idx, start:end, col],
                    dtype=longi_batch.dtype, device=longi_batch.device,
                )

        return np.asarray(_predict_for_explanation(
            explainer, longi_batch, t_eval,
            prediction_type=prediction_type, static_feat=static_batch,
        )).reshape(-1)

    background = np.repeat(bg_idx[:, None], n_intervals, axis=1)
    subject = np.full((1, n_intervals), sel_idx, dtype=int)
    shap_values = shap.KernelExplainer(wrapper_predict, background).shap_values(
        subject, l1_reg=0.0,
    )

    absolute = np.abs(np.asarray(shap_values)).reshape(-1)
    total = absolute.sum()
    return absolute / total if total > 1e-12 else np.full(n_intervals, 1 / n_intervals)


def _refine_feature_intervals(
    explainer, sel_idx, expl_longi, expl_static, col,
    pred_time_idx, bg_idx, pred_t, prediction_type,
    longi_dtype, delta_2, n_split,
):
    intervals = [(0, pred_time_idx)]
    interval_importance = np.ones(1, dtype=float)
    split_mask = np.ones(1, dtype=bool)

    for _ in range(1, n_split):
        intervals = _split_intervals(intervals, split_mask)
        interval_importance = _estimate_interval_importance(
            explainer, sel_idx, expl_longi, expl_static, col,
            intervals, bg_idx, pred_t, prediction_type, longi_dtype,
        )

        delta_2_current = min(delta_2, 1.0 / len(intervals))
        split_mask = np.array([
            score > delta_2_current and end - start > 1
            for score, (start, end) in zip(interval_importance, intervals)
        ])

        if not split_mask.any():
            break

    return intervals, interval_importance


# ============================================================
# DynSurvX: two-phase explanation for one individual
# ============================================================

def survshap_longi(
    explainer,
    sel_idx: int,
    expl_longi: np.ndarray,
    expl_static: np.ndarray,
    pred_time: np.ndarray | float,
    prediction_type: str = "survival",
    seed: int = 0,
    n_split: int = 3,
    delta_1: float = 0.2,
    delta_2: float = 0.25,
    reg: float = 5.0,
    intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate overall and interval-wise DynSurvX importance.

    delta_1: Fixed feature-screening threshold (default 0.2).
    delta_2: Phase 2 interval-splitting threshold (default 0.25).
    The entropy tolerance is hard-coded at 0.1.
    """
    rng = np.random.default_rng(seed)
    longi_dtype = expl_longi.dtype if torch.is_tensor(expl_longi) else torch.float32
    expl_longi = (
        expl_longi.detach().cpu().numpy()
        if torch.is_tensor(expl_longi) else np.asarray(expl_longi)
    )

    if expl_longi.ndim != 2:
        raise ValueError("expl_longi must have shape (n_sampling_times, n_features)")

    n_sampling_times, n_columns = expl_longi.shape
    n_longi_feats = n_columns - 1
    n_static_feats = 0 if expl_static is None else len(expl_static)
    n_features = n_longi_feats + n_static_feats

    if n_features == 0:
        raise ValueError("No feature is available for explanation")

    pred_t = float(np.asarray(pred_time).reshape(-1)[0])
    sampling_times = expl_longi[:, 0]

    if np.any(np.diff(sampling_times) < 0):
        raise ValueError("sampling_times must be sorted increasingly")

    if not sampling_times[0] <= pred_t <= sampling_times[-1]:
        raise ValueError(f"pred_time={pred_t} is outside the observation grid")

    # Preserve the original prediction-index convention.
    pred_time_idx = min(
        int(np.searchsorted(sampling_times, pred_t, side="right")) + 1,
        n_sampling_times,
    )
    replacement_mask = np.asarray(explainer.survival_labels)[:, 0] > pred_t + 1e-8

    if not replacement_mask.any():
        raise ValueError(f"No background individual survives beyond pred_time={pred_t}")

    # ---------- Phase 1: feature importance ----------
    feature_importance = _estimate_feature_importance(
        explainer, expl_longi, expl_static, pred_t, prediction_type,
        n_longi_feats, n_static_feats, replacement_mask, reg, intercept, rng,
    )
    is_concentrated = _should_refine(feature_importance, delta_1)

    # ---------- Phase 2: temporal refinement ----------
    importance = np.zeros((n_features, n_sampling_times), dtype=float)
    mask = np.zeros_like(importance)
    eligible_idx = np.flatnonzero(replacement_mask)
    bg_idx = rng.choice(eligible_idx, size=min(20, len(eligible_idx)), replace=False)

    for j in range(n_longi_feats):
        if is_concentrated and feature_importance[j] > delta_1:
            intervals, interval_scores = _refine_feature_intervals(
                explainer, sel_idx, expl_longi, expl_static, j + 1,
                pred_time_idx, bg_idx, pred_t, prediction_type,
                longi_dtype, delta_2, n_split,
            )

            for (start, end), score in zip(intervals, interval_scores):
                importance[j, start:end] = feature_importance[j] * score / (end - start)
                mask[j, start:end] = 1.0
        else:
            importance[j, :pred_time_idx] = feature_importance[j] / pred_time_idx
            mask[j, :pred_time_idx] = 1.0

    # ---------- Static features ----------
    for j in range(n_static_feats):
        idx = n_longi_feats + j
        importance[idx, :pred_time_idx] = feature_importance[idx] / pred_time_idx
        mask[idx, :pred_time_idx] = 1.0

    return importance, mask
