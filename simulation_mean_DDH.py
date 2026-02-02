import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

import warnings
warnings.filterwarnings('ignore')

from examples.learner.utils import score
from examples.data.longi_simulation import longi_sim
from src.survinsights.local_explaination._survlongishap import survshap_longi
from src.survinsights.longi_explainer import explainer
from examples.learner.dynamic_deephit_ext import construct_df, Dynamic_DeepHit_ext

n_longi_feats = 5
coefs = torch.Tensor([1, 0, 1., 0, 5., 0.])
coef_mask = torch.Tensor([1, 1, 1, 1, 1, 1])
important_feat_mask = torch.Tensor([0, 1, 0, 0, 0])
theta = torch.Tensor([0., -.3, 0., .2, 0.])
hurst_list = [0.9, 0.45, 0.85, 0.35, 0.8]
paths, surv_labels, static_feature, factor, ddh_info_sup, _, _ = longi_sim.Simulation().load(n_samples=1000, n_sampling_times = 200, end_time = 10, 
                                         dim=n_longi_feats + 1, intercept=True, threshold=12,
                                         coefs=coefs, coefs_mask=coef_mask, 
                                         important_feat_mask=important_feat_mask,
                                         hurst_list=hurst_list,
                                         theta=theta, n_static_feats=3)
important_feat_mask = (coefs != 0).int()
longi_feat_list = ['X'+ str(i+1) for i in range(n_longi_feats)]
n_samples, n_sampling_times, _ = paths.shape
sampling_times = paths[0, :, 0]
surv_times, surv_inds = surv_labels[:, 0], surv_labels[:, 1]
paths_init = paths.clone() # for MC true survival function


# Setup for experiment training
train_test_share = .8
n_samples = paths.shape[0]
n_train_samples = int(train_test_share * n_samples)
train_index = np.random.choice(n_samples, n_train_samples, replace=False)
test_index = [i for i in np.arange(n_samples) if i not in train_index]

paths_train = paths[train_index, :, :]
surv_labels_train = surv_labels[train_index, :]
if static_feature is not None:
    static_feature_train = static_feature[train_index, :]
else:
    static_feature_train = None

paths_test = paths[test_index, :, :]
surv_labels_test = surv_labels[test_index, :]
if static_feature is not None:
    static_feature_test = static_feature[test_index, :]
else:
    static_feature_test = None

cont_feat, bin_feat, time_scale, bin_df = ddh_info_sup
df = construct_df(paths.clone(), surv_labels, cont_feat, bin_feat, time_scale, bin_df)

dynamic_deephit = Dynamic_DeepHit_ext()
(data, time_, label), (mask1, mask2, mask3), (data_mi) = dynamic_deephit.preprocess(df, cont_feat, bin_feat)
dynamic_deephit.sampling_times = np.array(sampling_times)
dynamic_deephit.ddh_info_sup = ddh_info_sup

# split data
tr_data, te_data = data[train_index, :, :], data[test_index, :, :]
tr_data_mi, te_data_mi = data_mi[train_index, :, :], data_mi[test_index, :, :]
tr_time,te_time = time_[train_index, :], time_[test_index, :]
tr_label,te_label = label[train_index, :], label[test_index, :]
tr_mask1,te_mask1 = mask1[train_index, :, :], mask1[test_index, :, :]
tr_mask2,te_mask2 = mask2[train_index, :, :], mask2[test_index, :, :]
tr_mask3,te_mask3 = mask3[train_index, :], mask3[test_index, :]

tr_data_full = (tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3)

fig = plt.figure(layout="constrained", figsize=(16, 4))
axs = fig.subplot_mosaic(
    """
    AAAAABB
    """,
    gridspec_kw={
        "wspace": 0.2,
        "hspace": 0.5,
    },
)
for ax in axs.values():
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_edgecolor("black")
# train
dynamic_deephit.train(tr_data_full, is_trained=False, ax=axs["A"], ckpt_dir="./dynamic_deephit_ckpt/simulation")


# performance evaluation
sampling_times = np.array(paths[0, :, 0])
tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
quantile_pred_times = np.array([.1, .2, .4])
pred_times = np.quantile(np.array(tte), quantile_pred_times)
n_eval_times = 3
eval_times = []
for k in range(n_eval_times):
    eval_times.append(max(np.quantile(np.array(tte), quantile_pred_times + (k+1) * .05) - pred_times))
eval_times = np.array(eval_times)
# predict
pred_time_scale = (pred_times * time_scale).astype(int)
eval_time_scale = (eval_times * time_scale).astype(int)
ddh_surv_preds = dynamic_deephit.predict(te_data, te_data_mi, 
                                         pred_time_scale, eval_time_scale)

n_pred_times = len(pred_times)
n_eval_times = len(eval_times)
ddh_cindex = np.zeros((n_pred_times, n_eval_times))
for j in np.arange(n_pred_times):
    pred_time = pred_times[j]
    
    # remove individuals whose survival time less than prediction time
    surv_times_test = surv_labels_test[:, 0]
    surv_inds_test = surv_labels_test[:, 1]
    idx_sel = surv_times_test >= pred_time
    surv_times_ = surv_times_test[idx_sel] - pred_time
    surv_inds_ = surv_inds_test[idx_sel]
    surv_labels_ = np.array([surv_times_, surv_inds_]).T
    surv_preds_ = ddh_surv_preds[:, j][idx_sel]

    ddh_cindex[j] = score("c_index", surv_labels_, surv_labels_, 
                          surv_preds_, eval_times)

sns.heatmap(ddh_cindex, annot=True, annot_kws={"size": 10}, cmap="viridis", ax=axs["B"])
axs["B"].set_xticklabels([str(round(x, 2)) for x in eval_times])
axs["B"].set_yticklabels([str(round(x, 2)) for x in pred_times])
axs["B"].tick_params(axis='y', labelsize=12)
axs["B"].tick_params(axis='x', labelsize=12)
axs["B"].set_xlabel("$\delta_t$", fontweight="semibold", fontsize=15)
axs["B"].set_ylabel("$p_t$", fontweight="semibold", fontsize=15)
axs["B"].set_title("C_index $(p_t, \delta_t)$", fontweight="semibold", fontsize=15)
axs["B"].collections[0].colorbar.ax.tick_params(labelsize=12)
axs["B"].legend().remove()
plt.savefig('figures/simu_DDH_train.pdf', bbox_inches="tight")

quantile_pred_times = np.array([.5, .75])
tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
pred_times = np.quantile(np.array(tte), quantile_pred_times)
dt = sampling_times[1] - sampling_times[0]
# K = 200
all_shap = []
for pred_time in pred_times:
    sel_paths = paths[surv_labels[:, 0] > pred_time + 10 * dt]
    sel_static_feature = static_feature[surv_labels[:, 0] > pred_time + 10 * dt]
    sel_surv_labels = surv_labels[surv_labels[:, 0] > pred_time + 10 * dt]
    model_explainer = explainer(dynamic_deephit, "DDH", sel_paths, sel_surv_labels, sel_static_feature)
    seed = 0
    shap_res = []
    K = min(200, int(.75 * sel_paths.shape[0]))
    for i in range(K):
        tmp = survshap_longi(model_explainer, i, sel_paths[i], sel_static_feature[i], np.array([pred_time]), 
                            prediction_type="survival", seed=seed, n_split=4)

        seed += 1
        shap_res.append(tmp)

    all_shap.append(shap_res)

np.save('results/shap_simu_med_DDH.npy', np.array(all_shap, dtype=object), allow_pickle=True)