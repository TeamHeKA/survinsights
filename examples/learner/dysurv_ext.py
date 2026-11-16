import numpy as np
import torch

def make_dysurv_input(paths, static=None, pred_time=None, pad_value=-0.1):
    paths = np.asarray(paths, dtype=np.float32)
    times = paths[:, :, 0]
    x = paths[:, :, 1:].copy()

    if static is not None:
        static = np.asarray(static, dtype=np.float32)
        static_rep = np.repeat(static[:, None, :], x.shape[1], axis=1)
        x = np.concatenate([x, static_rep], axis=2)

    if pred_time is not None:
        valid = times <= pred_time

        if np.any(~valid.any(axis=1)):
            raise ValueError("No observations available before the landmark.")

        x[~valid] = pad_value

    return x.astype(np.float32)


def add_duration_channel(x, y):
    duration = np.broadcast_to(
        y[0][:, None, None],
        (*x.shape[:2], 1),
    )
    return np.concatenate([x, duration], axis=2).astype(np.float32)


# ============================================================
# Evaluation metrics
# ============================================================

def horizon_cindex(risk, duration, event, end_time):
    concordant, n_pairs = 0.0, 0

    for i in range(len(duration)):
        if event[i] != 1 or duration[i] > end_time:
            continue

        comparable = np.flatnonzero(duration > duration[i])

        n_pairs += len(comparable)
        concordant += np.sum(risk[i] > risk[comparable])
        concordant += 0.5 * np.sum(risk[i] == risk[comparable])

    return concordant / n_pairs if n_pairs else np.nan


def horizon_brier_score(risk, duration, event, end_time, censor_km, eps=1e-8):
    if censor_km is None:
        raise ValueError("censor_km is required for IPCW Brier score.")

    risk = np.asarray(risk, dtype=float)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event)

    observed_event = (duration <= end_time) & (event == 1)
    survived = duration > end_time

    g_t = max(float(censor_km.predict(end_time)), eps)
    g_event = np.maximum(
        np.asarray(censor_km.predict(duration), dtype=float), eps
    )

    weights = np.zeros(len(duration), dtype=float)
    weights[observed_event] = 1.0 / g_event[observed_event]
    weights[survived] = 1.0 / g_t

    outcome = observed_event.astype(float)

    return np.mean(weights * (outcome - risk) ** 2)


# ============================================================
# DySurv wrapper
# ============================================================

class DySurvWrapper:
    def __init__(self, model, time_col=0, feature_cols=None, padding_value=-0.1):
        self.model = model
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.padding_value = padding_value

        self.sampling_times = np.asarray(model.duration_index, dtype=float)
        self.device = next(model.net.parameters()).device
        self.model.net.eval()

    # ---------- Prepare longitudinal history ----------
    def _prepare_input(self, longi_feat, pred_time, static_feat=None):
        paths = np.asarray(longi_feat, dtype=np.float32)
        times = paths[:, :, self.time_col]

        cols = (
            [j for j in range(paths.shape[2]) if j != self.time_col]
            if self.feature_cols is None else list(self.feature_cols)
        )

        x = paths[:, :, cols].copy()
        valid = np.isfinite(times) & (times <= pred_time)

        if np.any(~valid.any(axis=1)):
            raise ValueError("No observations available before prediction time.")

        if static_feat is not None:
            static = np.asarray(static_feat, dtype=np.float32)
            static_rep = np.repeat(static[:, None, :], x.shape[1], axis=1)
            x = np.concatenate([x, static_rep], axis=2)

        x[~valid] = self.padding_value

        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # ---------- Predict unconditional survival ----------
    def _predict_survival(self, x):
        with torch.no_grad():
            mu, _ = self.model.net.encoder(x)
            logits = self.model.net.surv_net(mu)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1.0 - hazards, dim=1)

        return survival.cpu().numpy()

    # ---------- Conditional survival function ----------
    def sf(self, longi_feat, pred_times, static_feat=None):
        pred_times = np.atleast_1d(pred_times).astype(float)
        results = []

        for pred_time in pred_times:
            x = self._prepare_input(longi_feat, pred_time, static_feat)
            survival = self._predict_survival(x)

            s_landmark = np.array([
                np.interp(
                    pred_time, self.sampling_times, s,
                    left=1.0, right=s[-1],
                )
                for s in survival
            ])

            conditional = np.clip(
                survival / np.maximum(s_landmark[:, None], 1e-8),
                0.0, 1.0,
            )
            conditional[:, self.sampling_times <= pred_time] = 1.0

            results.append(conditional)

        return np.stack(results, axis=0)

    # ---------- Horizon-specific event risk ----------
    def risk(self, longi_feat, pred_time, horizon, static_feat=None):
        survival = self.sf(longi_feat, [pred_time], static_feat)[0]
        target = pred_time + horizon

        return 1.0 - np.array([
            np.interp(
                target, self.sampling_times, s,
                left=1.0, right=s[-1],
            )
            for s in survival
        ])

    # ---------- Dynamic performance evaluation ----------
    def score(
        self, longi_feat, surv_label, pred_time, eval_times,
        metric="c_index", static_feat=None, censor_km=None,
    ):
        labels = np.asarray(surv_label)
        duration, event = labels.T

        eval_times = np.atleast_1d(eval_times).astype(float)
        results = np.full(len(eval_times), np.nan)

        # Restrict evaluation to patients at risk at landmark
        eligible = duration > pred_time

        if eligible.sum() < 2:
            return results

        paths = np.asarray(longi_feat)[eligible]
        static = None if static_feat is None else np.asarray(static_feat)[eligible]

        duration, event = duration[eligible], event[eligible]

        for k, horizon in enumerate(eval_times):
            risk = self.risk(paths, pred_time, horizon, static)
            end_time = pred_time + horizon

            if metric == "c_index":
                results[k] = horizon_cindex(
                    risk, duration, event, end_time,
                )

            elif metric == "bs":
                results[k] = horizon_brier_score(
                    risk, duration, event, end_time, censor_km,
                )

            else:
                raise ValueError("metric must be 'c_index' or 'bs'.")

        return results
