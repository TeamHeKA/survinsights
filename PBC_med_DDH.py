import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

from examples.data.PBC import load_PBC_Seq
from examples.learner.dynamic_deephit_ext import construct_df, Dynamic_DeepHit_ext
from src.survinsights.longi_explainer import explainer
from src.survinsights.local_explaination._survlongishap import survshap_longi

import warnings
warnings.filterwarnings('ignore')


paths, surv_labels, longi_feat_list, static_feature, static_feature_list, ddh_info_sup = load_PBC_Seq.load()
n_samples, n_sampling_times, _ = paths.shape
n_longi_feats = paths.shape[-1] - 1
sampling_times = paths[0, :, 0]
surv_times, surv_inds = surv_labels[:, 0], surv_labels[:, 1]

cont_feat, bin_feat, time_scale, bin_df = ddh_info_sup
df = construct_df(paths.clone(), surv_labels, cont_feat, bin_feat, time_scale, bin_df)

dynamic_deephit = Dynamic_DeepHit_ext()
(data, time_, label), (mask1, mask2, mask3), (data_mi) = dynamic_deephit.preprocess(df, cont_feat, bin_feat)
dynamic_deephit.sampling_times = np.array(sampling_times)
dynamic_deephit.ddh_info_sup = ddh_info_sup

# Setup for experiment training
train_test_share = .8
n_samples = data.shape[0]
n_train_samples = int(train_test_share * n_samples)
train_index = np.random.choice(n_samples, n_train_samples, replace=False)
test_index = [i for i in np.arange(n_samples) if i not in train_index]

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
dynamic_deephit.train(tr_data_full, is_trained=False, ax=axs["A"], ckpt_dir="./dynamic_deephit_ckpt/PBC")


# performance evaluation
from examples.learner.utils import score
sampling_times = np.array(paths[0, :, 0])
tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
quantile_pred_times = np.array([.25, .5])
pred_times = np.quantile(np.array(tte), quantile_pred_times)
n_eval_times = 3
eval_times = []
for k in range(n_eval_times):
    eval_times.append(max(np.quantile(np.array(tte), quantile_pred_times + (k+1) * .15) - pred_times))
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
    surv_times_test = te_time.flatten()
    surv_inds_test = te_label.flatten()
    idx_sel = surv_times_test >= pred_time
    surv_times_ = surv_times_test[idx_sel] - pred_time
    surv_inds_ = surv_inds_test[idx_sel]
    surv_labels_ = np.array([surv_times_, surv_inds_]).T
    surv_preds_ = ddh_surv_preds[:, j][idx_sel]

    ddh_cindex[j] = score("c_index", surv_labels_, surv_labels_, 
                          surv_preds_, eval_time_scale)

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
plt.savefig('figures/PBC_DDH_train.pdf', bbox_inches="tight")

quantile_pred_times = np.array([.25, .5])
tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
pred_times = np.quantile(np.array(tte), quantile_pred_times)
dt = sampling_times[1] - sampling_times[0]
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
                            prediction_type="med-survival", seed=seed, n_split=4, intercept=True)

        seed += 1
        shap_res.append(tmp)

    all_shap.append(shap_res)

np.save('results/shap_PBC_med_DDH.npy', np.array(all_shap, dtype=object), allow_pickle=True)