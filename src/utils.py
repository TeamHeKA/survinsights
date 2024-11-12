import numpy as np

from scipy.interpolate import interp1d
from itertools import combinations
from scipy.stats import kstest
import pandas as pd
from sklearn.manifold import MDS


def convert_surv_label_structarray(surv_label):
    """
    Convert a normal array of survival labels to structured array. A structured
    array containing the binary event indicator as first field, and time of
    event or time of censoring as second field.

    Parameters
    ----------
    surv_label :  `np.ndarray`, shape=(n_samples, 2)
        Normal array of survival labels

    Returns
    -------
    surv_label_structarray : `np.ndarray`, shape=(n_samples, 2)
        Structured array of survival labels
    """
    surv_label_structarray = []
    n_samples = surv_label.shape[0]

    for i in range(n_samples):
        surv_label_structarray.append((bool(surv_label[i, 1]), surv_label[i, 0]))

    surv_label_structarray = np.rec.array(surv_label_structarray,
                                          dtype=[('indicator', bool),
                                                 ('time', np.float32)])

    return surv_label_structarray




def predict_survival_function_pycox(model, newdata):
    
    results=[]
        
    if len(newdata.shape)==1:
        surv=model.predict_surv_df(np.array([newdata]))
        t=np.array([surv.index])
        surv_prob=np.array([surv])
                 
        surv_func = interp1d(np.insert(t,0,0), np.insert(surv_prob,0,1), kind='previous', fill_value="extrapolate")
            
        results.append(surv_func)
                 
        return results
             
    elif len(newdata.shape)==2 and newdata.shape[0]==1:
        surv=model.predict_surv_df(newdata)
        t=np.array([surv.index])
        surv_prob=np.array([surv])
                 
        surv_func = interp1d(np.insert(t,0,0), np.insert(surv_prob,0,1), kind='previous', fill_value="extrapolate")
            
        results.append(surv_func)
                 
        return results
             
    else:
                 
        surv=model.predict_surv_df(newdata)
        t=np.array([surv.index])
                 
        surv_prob=np.array(surv)
                 
                 
        for i in range(0,surv_prob.shape[1],1):
            surv_prob_ind=surv_prob[:,i]
            surv_func = interp1d(np.insert(t,0,0), np.insert(surv_prob_ind,0,1), kind='previous', fill_value="extrapolate")
                     
            results.append(surv_func)
                 
        return results


def predict_cumulative_hazard_function_pycox(model, newdata):
    results=[]
    if len(newdata.shape)==1:
        results.append(-np.log(predict_survival_function_pycox(model,newdata)))
    elif len(newdata.shape)==2 and newdata.shape[0]==1:
        results.append(-np.log(predict_survival_function_pycox(model,newdata)))
    else:
        for i in range(newdata.shape[0]):
            results.append(-np.log(predict_survival_function_pycox(model,newdata[i,:])))
    
    return results



def feat_order(explainer, selected_features):
    data = explainer.data.copy(deep=True)
    encoder = explainer.encoders[selected_features]
    cate_features_ext = [feat for feat in data.columns.values if selected_features in feat]
    feat_values = encoder.inverse_transform(data[cate_features_ext]).flatten()
    data[selected_features] = feat_values
    group_values = np.unique(feat_values).tolist()
    group_comb = combinations(group_values, 2)
    n_groups = len(np.unique(group_values))
    dist_mat = np.zeros((n_groups, n_groups))
    for pair_feat in explainer.numeric_feats + explainer.cate_feats:
        if pair_feat not in selected_features:
            if pair_feat in explainer.numeric_feats:
                for pair in group_comb:
                    samp1 = data[data[selected_features] == pair[0]][pair_feat].values
                    samp2 = data[data[selected_features] == pair[1]][pair_feat].values
                    dist = kstest(samp2, samp1).statistic
                    idx1 = group_values.index(pair[0])
                    idx2 = group_values.index(pair[0])
                    dist_mat[idx1, idx2] += dist
                    dist_mat[idx2, idx1] += dist
            else:
                cate_features_ext = [feat for feat in data.columns.values if pair_feat in feat]
                encoder = explainer.encoders[pair_feat]
                feat_values = encoder.inverse_transform(data[cate_features_ext]).flatten()
                data[pair_feat] = feat_values
                for pair in group_comb:
                    samp1 = data[data[selected_features] == pair[0]][pair_feat]
                    samp2 = data[data[selected_features] == pair[1]][pair_feat]
                    samp1_tab = pd.crosstab(index=samp1, columns="count")
                    samp2_tab = pd.crosstab(index=samp2, columns="count")
                    samp1_dist = samp1_tab / samp1_tab.sum()
                    samp2_dist = samp2_tab / samp2_tab.sum()
                    dist = np.sum(np.abs(samp1_dist.values - samp2_dist.values))
                    idx1 = group_values.index(pair[0])
                    idx2 = group_values.index(pair[0])
                    dist_mat[idx1, idx2] += dist
                    dist_mat[idx2, idx1] += dist
    n_components = 1
    mds = MDS(n_components=n_components)
    dist_reduced = mds.fit_transform(dist_mat).flatten()
    dist_df = pd.DataFrame(np.array([np.array(group_values), dist_reduced]).T, columns=["groups", "dist"])
    dist_df = dist_df.sort_values(by=['dist'])
    encoder = explainer.encoders[selected_features]
    ordered_groups = encoder.transform(dist_df.groups.values.reshape((-1, 1))).toarray()

    return ordered_groups
