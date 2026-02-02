import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

import warnings
warnings.filterwarnings('ignore')

from examples.data.MSK import load_MSK
from src.survinsights.longi_explainer import explainer
from src.survinsights.local_explaination._survlongishap import survshap_longi

import warnings
warnings.filterwarnings('ignore')


paths, surv_labels, longi_feat_list, static_feature, static_feature_list, ddh_info_sup = load_MSK.load()
n_samples, n_sampling_times, _ = paths.shape
n_longi_feats = paths.shape[-1] - 1
sampling_times = paths[0, :, 0]
surv_times, surv_inds = surv_labels[:, 0], surv_labels[:, 1]

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

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_edgecolor('black')

plt.subplots_adjust(hspace=.35, wspace=0.3)

### TRAINING
from examples.learner.coxsig import CoxSignature
sig_order = 2
coxsig = CoxSignature(sig_level=sig_order, alphas=5e-2, max_iter=100)
for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_edgecolor('black')
coxsig.train(paths_train, surv_labels_train, static_feature_train, plot_loss=True, ax=axs[0])


### FEATURE IMPORTANCE
from itertools import product
def get_all_combinations(feat_lst):
    all_combinations = []
    for r in range(1, sig_order + 1):  # r varies from 1 to length of feat_lst
        all_combinations.extend([p for p in product(feat_lst, repeat=r)])
    return all_combinations

combinations_list = get_all_combinations(np.arange(n_longi_feats + 1).tolist())

coef_df = pd.DataFrame(columns=["Feat", "Type", "Val"])
pfi_list = []
for i in range(1, n_longi_feats + 1):
    fi = 0
    for j in range(2):
        fi += coxsig.model.coefs[combinations_list.index((i,) * (j + 1))]**2
    pfi_list.append(round(np.sqrt(fi), 3))

pfi_df = pd.DataFrame(np.array([longi_feat_list, ["Pure"] * len(longi_feat_list), pfi_list]).T, columns=["Feat", "Type", "Val"])
coef_df = pd.concat((coef_df, pfi_df))
n_static_feat = static_feature.shape[1]
for j in range(n_static_feat):
    coef_df.loc[len(coef_df)] = [static_feature_list[j], "Pure", (coxsig.model.coefs[-(n_static_feat - j)])**2]

cfi_list = []
for i in range(1, n_longi_feats + 1):
    fi = 0
    for j in range(len(combinations_list)):
        if i in combinations_list[j]:
            if (combinations_list[j] != (i,)) & (combinations_list[j] != (i, i ,)):
                fi += (coxsig.model.coefs[j])**2
    cfi_list.append(round(np.sqrt(fi), 3))

cfi_df = pd.DataFrame(np.array([longi_feat_list, ["Cross"] * len(longi_feat_list), cfi_list]).T, columns=["Feat", "Type", "Val"])
coef_df = pd.concat((coef_df, cfi_df))

coef_df["Val"] = pd.to_numeric(coef_df["Val"], errors="coerce")

sns.barplot(data=coef_df, x="Feat", y="Val", hue="Type", alpha=0.8, legend=True, ax=axs[1])
axs[1].set_xlabel("", fontweight="semibold", fontsize=15)
axs[1].tick_params(axis='x', labelsize=15, labelrotation=90)
axs[1].tick_params(axis='y', labelsize=15)
axs[1].set_ylabel("Feature Importance", fontweight="semibold", fontsize=15)
axs[1].legend(fontsize=12)


# setup
sampling_times = np.array(paths[0, :, 0])
tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
quantile_pred_times = np.array([.1, .25, .5])
pred_times = np.quantile(np.array(tte), quantile_pred_times)
n_eval_times = 3
eval_times = []
for k in range(n_eval_times):
    eval_times.append(max(np.quantile(np.array(tte), quantile_pred_times + (k+1) * .05) - pred_times))
eval_times = np.array(eval_times)

# evaluation
c_index = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'c_index', static_feat = static_feature_test)

sns.heatmap(c_index, annot=True, annot_kws={"size": 10}, cmap="viridis", ax=axs[2])
axs[2].set_xticklabels([str(round(x, 2)) for x in eval_times])
axs[2].set_yticklabels([str(round(x, 2)) for x in pred_times])
axs[2].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='x', labelsize=12)
axs[2].set_xlabel("$\delta_t$", fontweight="semibold", fontsize=15)
axs[2].set_ylabel("$p_t$", fontweight="semibold", fontsize=15)
axs[2].set_title("C_index $(p_t, \delta_t)$", fontweight="semibold", fontsize=15)
axs[2].collections[0].colorbar.ax.tick_params(labelsize=12)
axs[2].legend().remove()
plt.tight_layout()
plt.savefig('figures/MSK_Coxsig_train.pdf')

quantile_pred_times = np.array([.25, .5])
pred_times = np.quantile(np.array(tte), quantile_pred_times)
dt = sampling_times[1] - sampling_times[0]
all_shap = []
for pred_time in pred_times:
    sel_paths = paths[surv_labels[:, 0] > pred_time + 10 * dt]
    sel_static_feature = static_feature[surv_labels[:, 0] > pred_time + 10 * dt]
    sel_surv_labels = surv_labels[surv_labels[:, 0] > pred_time + 10 * dt]
    model_explainer = explainer(coxsig, "coxsig", sel_paths, sel_surv_labels, sel_static_feature)
    seed = 0
    shap_res = []
    K = min(200, int(.75 * sel_paths.shape[0]))
    for i in range(K):
        tmp = survshap_longi(model_explainer, i, sel_paths[i], sel_static_feature[i], pred_time=np.array([pred_time]), 
                                        prediction_type="med-survival", seed=seed, n_split=4, intercept=True, reg=.1)

        seed += 1
        shap_res.append(tmp)

    all_shap.append(shap_res)

np.save('results/shap_MSK_med_CoxSig.npy', np.array(all_shap, dtype=object), allow_pickle=True)