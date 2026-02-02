import pandas as pd
import numpy as np
import math
import seaborn as sns
import torch
sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')
import matplotlib.pyplot as plt
from src.survinsights.longi_prediction import predict, ddh_predict
# from src.survinsights.longi_prediction import predict
from itertools import product
from typing import Tuple, Sequence
import shap

def _shap_kernel_weight(s: int, m: int) -> float:
    """
    Shapley kernel for a coalition of size s in a universe of m features.
    Handles edge coalitions by returning 1 (as in your original code).
    """
    if s == 0 or s == m:
        return 1.0
    return (m - 1) / (math.comb(m, s) * s * (m - s))

def _soft_threshold(z, lam):
    """Soft-thresholding operator S(z, λ) = sign(z) * max(|z| - λ, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def weighted_lasso_coordinate_descent(A, b, lam=10., sample_weight=None,
                                      max_iter=1000, tol=1e-6, intercept=True):
    """
    Solve: 0.5 * (Ax - b)^T W (Ax - b) + lam * ||x||_1
    using coordinate descent (W = diag(sample_weight)).

    Parameters
    ----------
    A : np.ndarray, shape (n_samples, n_features)
        Design matrix.
    b : np.ndarray, shape (n_samples,)
        Target vector.
    lam : float
        L1 regularization strength λ (must be >= 0).
    sample_weight : np.ndarray or None, shape (n_samples,), optional
        Nonnegative weights w_i. If None, all weights are 1.
    max_iter : int
        Maximum number of coordinate descent iterations.
    tol : float
        L2-norm tolerance for convergence on x.

    Returns
    -------
    x : np.ndarray, shape (n_features,)
        Estimated coefficient vector.
    """
    A = np.asarray(A, dtype=float)
    if intercept:
        A = np.column_stack((A, np.ones(A.shape[0])))
    b = np.asarray(b, dtype=float)

    n, d = A.shape

    if sample_weight is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if w.shape[0] != n:
            raise ValueError("sample_weight must have length n_samples")

    # Precompute weighted normal-equation terms:
    # Q = A^T W A,  q = A^T W b
    # where W = diag(w)
    WA = w[:, None] * A           # n x d
    Q = A.T @ WA  + + 1e-6 * np.eye(d) # d x d
    q = A.T @ (w * b)             # d

    # Initialize coefficients
    x = np.zeros(d, dtype=float)

    for it in range(max_iter):
        x_old = x.copy()

        for j in range(d):
            # rho_j = q_j - sum_{k != j} Q_jk x_k
            # implement as: q_j - (Q_j· x) + Q_jj x_j
            Qj = Q[j, :]
            rho_j = q[j] - Qj @ x + Q[j, j] * x[j]

            # Soft-threshold and divide by Q_jj
            if Q[j, j] > 0:
                x[j] = _soft_threshold(rho_j, lam) / Q[j, j]
            else:
                x[j] = 0.0

        # Check convergence
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break
    if intercept:
        return x[:-1]
    else:
        return x

def _weighted_least_squares_lasso(coals: np.ndarray, w: np.ndarray, y: np.ndarray, lam: float, intercept: bool) -> np.ndarray:
    """
    Solve (C^T W C) beta = C^T W y for beta.
    coals: (K, p) coalition matrix in {0,1}
    w:     (K,) nonnegative weights
    y:     (K,) responses (predictions under each coalition)
    """
    return weighted_lasso_coordinate_descent(coals, y, lam=lam, sample_weight=w, intercept=intercept)


def _build_coalition_tensor_from_data(
    base: Tuple[torch.Tensor, torch.Tensor],                  # ((n_i, L), (d,)) for one subject/time grid
    time_col: int,                     # index of time column (0)
    longi_feat_col_indices: Sequence[int],   # which longitudinal columns are eligible
    coals: Tuple[np.ndarray, np.ndarray],                 # ((K, p), (K, d)) coalitions
    replacement: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Create a (K, n_i, L) tensor where selected longitudinal features are kept
    and unselected ones are zeroed (time column is always copied).
    """
    (expl_longi, expl_static) = base
    (longi_coals, static_coals) = coals
    (longi_replacement, static_replacement) = replacement

    if static_coals is not None:
        K, d = static_coals.shape
        # static_coal_data = np.zeros((K, d), dtype=expl_static.dtype)
        static_coal_data = np.tile(static_replacement[np.random.randint(0, static_replacement.shape[0], (1,))], (K, 1))
        for k in range(K):
            for j in range(d):
                if static_coals[k, j] == 1:
                    static_coal_data[k, j] = expl_static[j]
    else:
        static_coal_data = None

    K, p = longi_coals.shape
    n_i, L = expl_longi.shape
    # longi_coal_data = torch.zeros((K, n_i, L), dtype=expl_longi.dtype)
    longi_coal_data = longi_replacement[torch.randint(0, longi_replacement.size(0), (1,))].repeat(K, 1, 1)
    longi_coal_data[:, :, time_col] = expl_longi[:, time_col]  # keep time
    # map each coalition column to its feature index in base
    for k in range(K):
        for j, f_idx in enumerate(longi_feat_col_indices):
            if longi_coals[k, j] == 1:
                longi_coal_data[k, :, f_idx] = expl_longi[:, f_idx]
            else:
                longi_coal_data[k, :, f_idx] = 0.0

    return longi_coal_data, static_coal_data


def _choose_explanation_times(
    sampling_times: np.ndarray,
    pred_time: float,
    n_times: int,
    rng: np.random.Generator,
    guard: int = 3,
) -> np.ndarray:
    """
    Pick 'n_times' interior times strictly before pred_time, avoid the first/last
    'guard' indices to reduce boundary artifacts. Append pred_time at the end and
    return a sorted array.
    """
    mask = (sampling_times < pred_time)
    idx = np.flatnonzero(mask)
    if len(idx) <= 2 * guard:
        # Fallback: just use a few earliest times < pred_time
        candidates = sampling_times[idx]
    else:
        candidates = sampling_times[idx][guard:-guard]

    candidates = np.unique(candidates)
    n_pick = min(n_times, max(1, len(candidates)))
    picked = np.sort(rng.choice(candidates, size=n_pick, replace=False))
    # picked = np.sort(np.random.choice(candidates, size=n_pick, replace=False))

    return np.concatenate([picked, np.array([pred_time], dtype=float)])

def _split_intervals(intervals: Sequence[Tuple[int, int]], split_mask: np.ndarray) -> list[Tuple[int, int]]:
    """
    Your original 'split_intervals' logic (assumed) – here we implement a simple
    bisection of intervals where split_mask == 1; otherwise keep interval.
    intervals: list of (start, end) inclusive-exclusive (end not included).
    """
    new_intervals: list[Tuple[int, int]] = []
    for (start, end), do_split in zip(intervals, split_mask.astype(bool)):
        if not do_split or end - start <= 1:
            new_intervals.append((start, end))
        else:
            mid = (start + end) // 2
            new_intervals.extend([(start, mid), (mid, end)])
    return new_intervals


def survshap_longi(
    explainer,
    sel_idx: int,
    expl_longi: np.ndarray,
    expl_static: np.ndarray,
    pred_time: np.ndarray | float,
    prediction_type: str = "survival",
    seed: int = 0,
    n_split: int = 3,
    n_times: int = 5,
    importance_threshold: float = 0.1,
    split_threshold: float = 0.4,
    reg: float = 5.,
    intercept: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SurvSHAP


    Parameters
    ----------
    explainer : Python object
        Instance of the explainer class for the survival model.
    new_data : pd.DataFrame or np.ndarray
        Data for new observations whose predictions need to be explained.
    sample_id : int, optional
        Index of the observation in new_data to explain. Defaults to the first observation (index 0).

    Returns
    -------
    pd.DataFrame
        DataFrame containing survshap values for each feature and time point.
    """

    rng = np.random.default_rng(seed)

    # --- Validate & unpack shapes
    if expl_longi.ndim != 2:
        raise ValueError("new_data must have shape (n_i, L)")
    n_sampling_times, L = expl_longi.shape

    # Normalize pred_time
    if isinstance(pred_time, (list, tuple, np.ndarray)):
        pred_t = float(np.asarray(pred_time).ravel()[0])
    else:
        pred_t = float(pred_time)

    # --- Time grid & index at prediction time
    sampling_times = expl_longi[:, 0]
    dt = sampling_times[1] - sampling_times[0]
    eps = 1e-8
    # we search in (1:), and subtract 1 like your original code; protect lower bound
    pred_time_idx = np.searchsorted(sampling_times[1:], pred_t + eps).item() - 1
    pred_time_idx = max(1, min(pred_time_idx, n_sampling_times - 1))

    # --- Feature setup
    time_col = 0
    longi_feat_cols = list(range(1, L))                 # longitudinal feature columns
    n_longi_feats = len(longi_feat_cols)                # number of longi features
    if expl_static is not None:
        n_static_feats = len(expl_static)
    all_coals = np.array(list(product([0, 1], repeat=n_longi_feats + n_static_feats)))
    K = len(all_coals)
    sel_coals_idx = rng.choice(len(all_coals), size=K, replace=False)
    data_coals = all_coals[sel_coals_idx]
    longi_coals = data_coals[:, :L-1]
    if expl_static is not None:
        static_coals = data_coals[:, L-1:]
    else:
        static_coals = None
        
    # Weights per coalition (Shapley kernel)
    w = np.array([_shap_kernel_weight(int(c.sum()), n_longi_feats + n_static_feats) for c in data_coals], dtype=float)

    # --- Build coalition tensors (K, n_i, L) for the chosen subject
    sel_id = explainer.survival_labels[:, 0] > pred_t + eps
    longi_coal_tensor, static_coal_array = _build_coalition_tensor_from_data(
        base=(expl_longi, expl_static),
        time_col=time_col,
        longi_feat_col_indices=longi_feat_cols,
        coals=(longi_coals, static_coals),
        replacement=(explainer.paths[sel_id], explainer.static_feats[sel_id])
    )

    # --- Explanation times (interior + pred_t at the end)
    exp_times = _choose_explanation_times(
        sampling_times=sampling_times,
        pred_time=pred_t,
        n_times=n_times,
        rng=rng,
        guard=3,
    )  # length n_times+1 (last is pred_t)
    # --- Aggregate importance across explanation times via time integral
    feat_importance = np.zeros(n_longi_feats + n_static_feats, dtype=float)

    n_trials = 3
    for k in range(n_trials):
        lam = reg * (0.75**k)
        for j in range(len(exp_times) - 1):
            t_eval = np.array([exp_times[j]], dtype=float)
            # Predict under all coalitions at time t_eval
            # y = predict(explainer, longi_coal_tensor, t_eval, prediction_type=prediction_type, static_feat = static_coal_array)
            if explainer.model_name == "DDH":
                y = ddh_predict(explainer, longi_coal_tensor, t_eval, prediction_type=prediction_type, static_feat=static_coal_array)
            else:
                y = predict(explainer, longi_coal_tensor, t_eval, prediction_type=prediction_type, static_feat=static_coal_array)

            y = np.asarray(y).reshape(-1)  # (K,)
            beta = _weighted_least_squares_lasso(data_coals, w, y, lam, intercept)  # (p,)
            feat_importance += np.abs(beta) * (exp_times[j + 1] - exp_times[j])

        # --- Normalize to percentage & compute entropy
        if np.any(feat_importance):
            feat_perc = np.abs(feat_importance) / np.sum(np.abs(feat_importance))
            break
        else:
            feat_perc = np.full(n_longi_feats + n_static_feats, 1.0 / (n_longi_feats + n_static_feats))
            pass
    print(feat_perc)
    uniform_entropy = np.log(n_longi_feats + n_static_feats)
    entropy = -np.sum(feat_perc * np.log(feat_perc + 1e-12))

    # Thresholds for “important” features
    perc_threshold = 1.0 / (n_longi_feats + n_static_feats)

    importance_res = np.zeros((n_longi_feats + n_static_feats, n_sampling_times), dtype=float)
    importance_mask = np.zeros((n_longi_feats + n_static_feats, n_sampling_times), dtype=float)

    # --- If importance is concentrated, perform time-wise splitting for important features
    # is_concentrated = (entropy < uniform_entropy - importance_threshold) or (feat_perc > 1.5 * perc_threshold).any()
    alpha_1 = 1.4
    is_concentrated = (entropy < uniform_entropy - importance_threshold) or (feat_perc > alpha_1 * perc_threshold).any()

    if is_concentrated:
        # For each feature, potentially split the time axis adaptively
        for f_idx_in_p, col in enumerate(longi_feat_cols):  # f_idx_in_p in [0..p-1], col in [1..L-1]
            # if feat_perc[f_idx_in_p] > perc_threshold:
            if feat_perc[f_idx_in_p] > alpha_1 * perc_threshold:
                # Start with one interval [0, pred_time_idx)
                intervals = [(0, pred_time_idx)]
                split_mask = np.ones(1)  # force one round of evaluation

                for _ in range(1, n_split):
                    intervals = _split_intervals(intervals, split_mask)
                    n_intervals = len(intervals)
                    def wraper_predict(x):
                        coal_tensor_int = torch.zeros((x.shape[0], n_sampling_times, n_longi_feats + 1), dtype=expl_longi.dtype)
                        coal_tensor_int[:] = expl_longi  # copy everything
                        static_feat_int = np.zeros((x.shape[0], n_static_feats), dtype=expl_static.dtype)
                        static_feat_int[:] = expl_static

                        for i, x_ in enumerate(x):
                            for j in range(len(x_)):
                                coal_tensor_int[i, intervals[j][0]:intervals[j][1], col] = explainer.paths[x_[j], intervals[j][0]:intervals[j][1], col]
                        if explainer.model_name == "DDH":
                            pred = ddh_predict(explainer, coal_tensor_int, np.array([pred_t - 4 * dt]), prediction_type=prediction_type, static_feat=static_feat_int)
                        else:
                            pred = predict(explainer, coal_tensor_int, np.array([pred_t - 4 * dt]), prediction_type=prediction_type, static_feat=static_feat_int)

                        return pred

                    n_bg = 20
                    bg = np.array([[i + explainer.paths.shape[0] - n_bg] * n_intervals for i in range(20)])
                    shap_explainer = shap.KernelExplainer(wraper_predict, bg)
                    shap_values = shap_explainer.shap_values(np.array([[sel_idx] * n_intervals]), l1_reg=0.)
                    perc_int = (np.abs(shap_values) / np.sum(np.abs(shap_values))).flatten()

                    # Decide which intervals to split next round
                    alpha_2 = .25
                    split_threshold_int = min(alpha_2, 1 / n_intervals)
                    split_mask = (perc_int > split_threshold_int).astype(float)

                    if np.any(split_mask):
                        pass
                    else:
                        break

                # Distribute final interval importances across time indices
                for j, (start, end) in enumerate(intervals):
                    length = max(1, end - start)
                    importance_res[f_idx_in_p, start:end] = (perc_int[j] * feat_perc[f_idx_in_p]) / length
                    importance_mask[f_idx_in_p, start:end] = 1.0
            else:
                # Not “important”: spread uniformly up to pred_time_idx
                length = max(1, pred_time_idx)
                importance_res[f_idx_in_p, :pred_time_idx] = feat_perc[f_idx_in_p] / length
                importance_mask[f_idx_in_p, :pred_time_idx] = 1.0
    else:
        # Not concentrated: spread feature importances uniformly up to pred_time_idx
        length = max(1, pred_time_idx)
        importance_res[:, :pred_time_idx] = (feat_perc[:, None]) / length
        importance_mask[:, :pred_time_idx] = 1.0

    # static feature importance
    length = max(1, pred_time_idx)
    for j in range(n_static_feats):
        importance_res[j + n_longi_feats, :pred_time_idx] = (feat_perc[j + n_longi_feats]) / length
        importance_mask[j + n_longi_feats, :pred_time_idx] = 1.0

    return importance_res, importance_mask