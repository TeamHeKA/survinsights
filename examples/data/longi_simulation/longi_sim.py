import numpy as np
import torch
from stochastic.processes.continuous import FractionalBrownianMotion
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

rng = np.random.default_rng(42)

def compute_factor(
    T: int,
    delta_t: float,
    coef_list: np.ndarray,
    theta_list: np.ndarray,
    static_feature: np.ndarray,
    static_feature_factor: np.ndarray,
    params,
    paths
) -> np.ndarray:
    """
    Compute factor matrix for dynamic and static features.
    """
    mu, sigma, _ = params
    n_time = T
    L = len(coef_list)
    factors = []

    # dynamic contributions
    for l, coef in enumerate(coef_list):
        factor_l = []
        M = 0
        for t in range(n_time):
            if t == n_time - 1:
                factor_l.append(coef)
            else:
                decay =  (theta_list[l] * delta_t)**(int(t != 0)) * ((1 - theta_list[l] * delta_t) ** (n_time - t - 2))
                factor_l.append(coef * decay)
            rnd = rng.normal(0, 0.1)
            M += ((1 - theta_list[l] * delta_t) ** (n_time - t - 1)) * (theta_list[l] * delta_t * mu + sigma * (sigma / L) * np.sqrt(delta_t) * rnd)

        factors.append((paths[:, :n_time, l + 1]).numpy() * np.array(factor_l))
    # static contributions
    for d, val in enumerate(static_feature_factor):
        factors.append(np.array([(static_feature[:, d] * val) / n_time] * n_time).T.reshape(factors[0].shape))

    return np.array(factors), M.numpy()


class Simulation:
    """
    Generate survival data with:
    - Longitudinal features (fractional Brownian motion paths).
    - Survival times defined by when a latent trajectory crosses a threshold.
    """

    # ------------------------------------------------------------------
    # Longitudinal data
    # ------------------------------------------------------------------
    def get_path(
        self,
        n_samples: int,
        n_times: int,
        end_time: float,
        dim: int = 3,
        intercept: bool = True,
        important_feat_mask: np.ndarray | None = None,
        hurst_list: list[float] | None = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Simulate longitudinal features as sample paths of fBM.
        """
        step = end_time / n_times
        sampling_times = torch.arange(0, end=end_time, step=step)
        n_times = sampling_times.shape[0]

        paths = torch.empty(n_samples, n_times, dim)
        paths[:, :, 0] = sampling_times

        # fBM generators
        generators = [
            FractionalBrownianMotion(t=end_time, hurst=h, rng=rng)
            for h in hurst_list
        ]

        # simulate each path
        for j in range(n_samples):
            for d, gp in enumerate(generators):
                paths[j, :, d + 1] = torch.tensor(gp.sample(n_times - 1))

        # optional intercept
        if intercept:
            torch.manual_seed(0)
            intercepts = torch.randn_like(paths[:, 0, 1:])
            intercepts = intercepts.repeat_interleave(n_times, dim=0)
            paths[:, :, 1:] += (
                intercepts.reshape(n_samples, -1, dim - 1) * important_feat_mask
            )

        return paths, np.array(sampling_times)

    # ------------------------------------------------------------------
    # Static features
    # ------------------------------------------------------------------
    def get_static_feature(
        self,
        n_samples: int,
        n_features: int = 30,
        active_perc: float = 0.7,
        cov_corr: float = 0.5,
        dtype: str = "float64",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate Gaussian static features with Toeplitz covariance.
        """
        if n_features <= 0:
            return None, None

        cov = toeplitz(cov_corr ** np.arange(0, n_features))
        feats = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

        n_active_feats = int(active_perc * n_features)
        coef = np.zeros(n_features)
        coef[:n_active_feats] = np.random.uniform(low=-5, high=5, size=n_active_feats)
        if not np.any(np.abs(coef) > 4):
            coef[0] = -5.
        return feats.astype(dtype), coef

    # ------------------------------------------------------------------
    # Trajectories & survival times
    # ------------------------------------------------------------------
    def get_trajectory(
        self,
        path: torch.Tensor,
        coefs: np.ndarray,
        theta: np.ndarray,
        static_feat: np.ndarray,
        static_coef: np.ndarray,
        params
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Simulate latent trajectory using Ornstein-Uhlenbeck SDE.
        """
        # mu, sigma, kappa = 0.1, 1.0, 0.5
        mu, sigma, kappa = params
        n_samples, n_times, _ = path.shape
        traj = torch.ones((n_samples, n_times))
        time_step = path[0, 1, 0] - path[0, 0, 0]

        L = path.shape[-1] - 1
        for i in range(n_samples):
            prev = torch.ones(L)
            for t in range(1, n_times):
                delta = path[i, t, :] - path[i, t - 1, :]
                # delta = path[i, t - 1, :] * time_step
                rnd = rng.normal(0, 0.1)
                tmp = (
                    prev
                    + coefs[1:] * delta[1:]
                    - theta * (prev - mu / L) * time_step
                    + (sigma / L) * np.sqrt(time_step) * rnd
                    + time_step / L
                )
                static_term = 0
                if static_feat is not None and static_coef is not None:
                    static_term = kappa * (static_feat[i] * static_coef).sum()
                traj[i, t] = tmp.sum() + static_term
                prev = tmp

        if static_coef is not None:
            static_feature_factor = static_coef * kappa
        else:
            static_feature_factor = np.array([])

        return traj, static_feature_factor

    def get_survival_label(
        self,
        paths: torch.Tensor,
        coefs: np.ndarray,
        end_time: float,
        threshold: float = 2.5,
        theta: np.ndarray | None = None,
        static_feat: np.ndarray | None = None,
        static_coef: np.ndarray | None = None,
        params = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute survival labels from trajectories crossing threshold.
        """
        traj, static_factor = self.get_trajectory(paths, coefs, theta, static_feat, static_coef, params)
        max_paths = traj - threshold

        idxs = np.argmax(max_paths > 0, axis=1)
        times_ext = np.array(paths[:, :, 0])
        surv_times = np.take(times_ext, idxs)
        surv_times[surv_times == 0] = end_time

        surv_inds = idxs != 0
        return np.array(surv_times), np.array(surv_inds), static_factor, traj

    # ------------------------------------------------------------------
    # Dataset generator
    # ------------------------------------------------------------------
    def generate_simulated_dataset(
        self,
        n_samples: int,
        n_times: int,
        end_time: float,
        dim: int = 3,
        intercept: bool = True,
        threshold: float = 5,
        coefs: np.ndarray | None = None,
        coefs_mask: np.ndarray | None = None,
        important_feat_mask: np.ndarray | None = None,
        hurst_list: list[float] | None = None,
        theta: np.ndarray | None = None,
        n_static_feats: int = 5,
        params=None
    ):
        """
        Full pipeline to simulate longitudinal + static features and survival times.
        """
        paths, times = self.get_path(
            n_samples, n_times, end_time, dim, intercept,
            important_feat_mask=important_feat_mask, hurst_list=hurst_list
        )
        static_feat, static_coef = self.get_static_feature(n_samples, n_static_feats)
        surv_times, surv_inds, static_factor, traj = self.get_survival_label(
            paths, coefs, end_time, threshold, theta, static_feat, static_coef, params
        )
        return paths, times, surv_times, surv_inds, static_feat, static_factor, traj

    def _truncate_paths_at_events(
        self,
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


    def _compute_factor_panel(
        self,
        paths: torch.Tensor,
        sampling_times: np.ndarray,
        coefs: np.ndarray,
        theta: np.ndarray,
        static_feature: np.ndarray,
        static_feature_factor: np.ndarray,
        # panel_T_cap: int = 1000,
        params = None
    ) -> np.ndarray:
        """
        Build factor panel: average (over T) normalized absolute factors.
        Returns: factor array of shape (n_features_total, n_times-1).
        """
        time_step = float(paths[0, 1, 0] - paths[0, 0, 0])
        n_times = sampling_times.shape[0]
        n_longi = paths.shape[-1] - 1
        n_samples = paths.shape[0]
        n_static = len(static_feature_factor) if static_feature_factor is not None else 0
        n_feat_total = n_longi + n_static

        # time grid excludes t=0 to match your previous np.arange(1, ...)
        time_grid = np.arange(1, n_times)
        # Factor tensor: (features, time_index, max_T_clip) filled with NaN
        factor_acc = np.full((n_feat_total, len(time_grid), n_times, n_samples), np.nan, dtype=float)

        for ti, T in enumerate(time_grid):  # T is number of steps (1..n_times-1)
            fac, C = compute_factor(
                T=T,
                delta_t=time_step,
                coef_list=coefs[1:],        # exclude time column
                theta_list=theta,
                static_feature=static_feature,
                static_feature_factor=static_feature_factor if n_static > 0 else [],
                params=params,
                paths=paths,
            )  # shape: (n_feat_total, T)

            # normalize by L1 (absolute)
            denom = np.abs(fac).sum(axis=0).sum(axis=-1) + np.abs(C)
            if denom.sum() > 0:
                fac_norm = np.swapaxes(np.abs(fac),1,2) / denom
            else:
                fac_norm = np.abs(fac)

            # store into accumulator (clip to panel_T_cap columns)
            T_clip = min(fac_norm.shape[1], n_times)
            factor_acc[:, ti, :T_clip] = fac_norm[:, :T_clip]

        return factor_acc

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize(
        self,
        paths: torch.Tensor,
        static_feature: np.ndarray,
        truncated_paths: torch.Tensor,
        surv_labels: np.ndarray,
        factor: np.ndarray,
        sel_idx: int = 1,
        save_path: str | None = "figures/simu_data.pdf",
        show: bool = True,
    ) -> None:
        """
        Visualize subject paths, feature importance, and survival histogram.
        """
        fig = plt.figure(layout="constrained", figsize=(16, 4))
        axs = fig.subplot_mosaic(
            """
            AABBBDD
            AACCCDD
            """,
            # gridspec_kw={
            #     "wspace": 0.5,
            #     "hspace": 0.5,
            # },
        )
        for ax in axs.values():
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_edgecolor("black")

        colors = ["r", "g", "b", "c", "m", "#E69F00", "#5E3C99", "#ff9896"]
        times = paths[sel_idx, :, 0]
        n_samples = paths.shape[0]
        n_longi = paths.shape[-1] - 1
        n_sampling_times = paths.shape[1]
        n_static_feats = static_feature.shape[1]
        dt = times[1] - times[0]
        surv_times, surv_inds = surv_labels[:, 0], surv_labels[:, 1]

        # ---- longitudinal paths
        for l, ts in enumerate(truncated_paths[sel_idx, :, 1:].T):
            axs['A'].plot(times, ts, color=colors[l], label=f"$X_{{{l+1}}}(t)$")
        sampling_times = paths[sel_idx,:,0]
        time_step = sampling_times[1] - sampling_times[0]
        axs['A'].set_title("Longitudinal features", fontweight="semibold", fontsize=15)
        axs['A'].legend(ncols=2, fontsize=10, loc="upper left")
        axs['A'].set_xlabel("") 
        axs['A'].tick_params(axis='x', labelsize=12) 
        axs['A'].tick_params(axis='y', labelsize=12) 
        axs['A'].set_ylabel("") 
        axs['A'].set_xlim(min(sampling_times), max(sampling_times) + 10 * time_step)

        # # ---- FEATURE IMPORTANCE
        pred_time_idx = 70
        factor_ext = []
        for i in range(n_samples):
            if surv_times[i] > float(pred_time_idx * dt):
                factor_ext.append(factor.T[i][:, pred_time_idx-1:pred_time_idx])
        factor_ = np.nanmean(np.array(factor_ext).T, axis=1).mean(axis=-1)

        k = 10
        factor_sel_avg = (
            factor_
            .reshape(factor_.shape[0], -1, k)
            .mean(axis=2, keepdims=True)
            .repeat(k, axis=2)
            .reshape(factor_.shape[0], n_sampling_times)
        )
        factor_sel_avg[factor_sel_avg == 0] = 1e-6

        factor_per_feat = factor_sel_avg[:, :pred_time_idx].sum(axis=1)
        x_labels = ["$X_" + str(i) + "$" for i in range(n_longi)] + ["$W_" + str(i) + "$" for i in range(n_static_feats)]

        score_df = pd.DataFrame(np.array([x_labels, factor_per_feat.tolist()]).T, columns = ["feat", "value"])
        score_df["value"] = score_df["value"].astype(float)
        sns.barplot(data=score_df, x="feat", y="value", ax=axs['B'], palette=colors)
        axs['B'].tick_params(axis='x', labelsize=12) 
        axs['B'].tick_params(axis='y', labelsize=12)
        axs['B'].set_xlabel("")
        axs['B'].set_ylabel("")
        axs['B'].set_title("Contribution power", fontweight="semibold", fontsize=15) 

        # ---- LONGITUDINAL feature importance
        for l, f in enumerate(factor_sel_avg[:n_longi]):
            label = f"$\\hat{{\\phi}}_{{X^{l+1}}}(t)$"
            axs['C'].plot(times[: len(f)][:pred_time_idx], f[:pred_time_idx], label=label, color=colors[l % len(colors)], linestyle="-")
        axs['C'].legend(ncols=3, fontsize=8, loc="upper left")
        axs['C'].tick_params(axis='x', labelsize=12) 
        axs['C'].tick_params(axis='y', labelsize=12)

        # ---- survival histogram
        axs['D'].hist(surv_times, bins=100, density=True)
        cens_pct = (1 - surv_inds.sum() / len(surv_inds)) * 100
        axs['D'].legend([Line2D([0], [0], lw=2)], [f"Censoring: {cens_pct:.1f}%"], loc="upper right", fontsize=10)
        axs['D'].set_title("Histogram of survival time", fontweight="semibold", fontsize=15)
        axs['D'].set_xlabel("") 
        axs['D'].tick_params(axis='x', labelsize=12) 
        axs['D'].tick_params(axis='y', labelsize=12) 
        axs['D'].set_ylabel("") 
        axs['D'].set_xlim(min(sampling_times), max(sampling_times) + 10 * time_step)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def load(
        self,
        n_samples: int = 500,
        n_sampling_times: int = 1000,
        end_time: float = 10.0,
        dim: int = 5,
        intercept: bool = True,
        threshold: float = 2.5,
        coefs: np.ndarray | None = None,
        coefs_mask: np.ndarray | None = None,          # kept for API parity
        important_feat_mask: np.ndarray | None = None,
        hurst_list: list[float] | None = None,
        theta: np.ndarray | None = None,
        n_static_feats: int = 5,
        save_path: str | None = "figures/simu_data.pdf",
        show: bool = True,
    ):
        """
        End-to-end simulation pipeline:
        1) Simulate longitudinal & static features and survival labels.
        2) Truncate paths at event times (freeze after event).
        3) Compute factor panel for importance visualization.
        4) Plot and return processed arrays.

        Returns
        -------
        truncated_paths : torch.Tensor, shape (n_samples, n_sampling_times, dim)
        surv_labels     : np.ndarray,   shape (n_samples, 2)  [time, event_ind]
        static_feature  : np.ndarray or None
        """
        # 1) simulate
        mu, sigma, kappa = 0.1, 1.0, 0.5
        params = (mu, sigma, kappa)
        paths, sampling_times, surv_times, surv_inds, static_feature, static_feature_factor, traj = \
            self.generate_simulated_dataset(
                n_samples=n_samples,
                n_times=n_sampling_times,
                end_time=end_time,
                dim=dim,
                intercept=intercept,
                threshold=threshold,
                coefs=coefs,
                coefs_mask=coefs_mask,
                important_feat_mask=important_feat_mask,
                hurst_list=hurst_list,
                theta=theta,
                n_static_feats=n_static_feats,
                params=params
            )

        surv_labels = np.column_stack([surv_times, surv_inds])

        # 2) truncate paths after event
        truncated_paths = self._truncate_paths_at_events(paths, sampling_times, surv_labels)

        # 3) compute factor panel
        if coefs is None or theta is None:
            raise ValueError("`coefs` and `theta` must be provided to compute factors.")
        factor = self._compute_factor_panel(
            paths=paths,
            sampling_times=sampling_times,
            coefs=coefs,
            theta=theta,
            static_feature=static_feature,
            static_feature_factor=static_feature_factor,
            params=params
        )

        # 4) visualize
        self.visualize(
            paths=paths,
            static_feature=static_feature,
            truncated_paths=truncated_paths,
            surv_labels=surv_labels,
            factor=factor,
            sel_idx=1,
            save_path=save_path,
            show=show,
        )

        # dynamic deephit info for preprocessing
        time_scale = n_sampling_times / end_time
        cont_feat = ["X_" + str(i) for i in range(dim - 1)]
        bin_feat = ["W_" + str(i) for i in range(n_static_feats)]
        bin_df = pd.DataFrame(data=static_feature, columns=bin_feat)
        bin_df["id"] = np.arange(n_samples)
        ddh_info_sup = (cont_feat, bin_feat, time_scale, bin_df)

        return truncated_paths, surv_labels, static_feature, factor, ddh_info_sup, traj, paths