# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:51:19 2024

@author: ducro
"""
import numpy as np
import pandas as pd
from src import performance

def counterfactual_perturbation_feature_importance(explainer, feats, surv_labels, eval_times=None, loss="brier_score", type="ratio" , variables = None):
    """
	Compute counterfactual perturbatiob feature importance (CPFI)


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model

	Returns
	-------
	CPFI_df : `np.ndarray`, shape=()
		Returns the CPFI value of selected features
	"""     
    if loss == "brier_score":
        bs_perf = performance.evaluate(explainer, feats, surv_labels,
		                               times=eval_times, metric="brier_score")["perf"].values

        feats_name = feats.columns.values.tolist()
        feat_importance_df_cols = ["feat", "times", "perf"]
        feat_importance_df = pd.DataFrame(columns=feat_importance_df_cols)
        n_eval_times  = len(eval_times)

        feats_name_org = explainer.numeric_feats + explainer.cate_feats
        feats_name_ext = explainer.numeric_feats
        
        for cate_feat_name in explainer.cate_feats:
            cate_feat_name_list = []
            for feat_name in feats_name:
                if cate_feat_name in feat_name:
                    cate_feat_name_list.append(feat_name)
            if len(cate_feat_name_list):
                feats_name_ext.append(cate_feat_name_list)
    
    
        for i in range(len(feats_name_ext)):
            feat_name = feats_name_ext[i]
            feat_name_org = feats_name_org[i]
            bs_perf_perm = np.zeros(n_eval_times)
            
            
            ### add perturbation here
            
            ## categorical feature with OneHotEncoding or Dummies
            if len(feat_name)>1:
                feats_perm_df = categorical_perturbation()
                bs_perf_perturbated = performance.evaluate(explainer, feats_perm_df, surv_labels,times=eval_times, metric="brier_score")["perf"].values
            
            ## categorical feature encoded on one variable
            elif (len(feat_name)==1) and (feat_name in explainer.cate_feats):
                feats_perm_df = unique_categorical_perturbation()
                bs_perf_perturbated = performance.evaluate(explainer, feats_perm_df, surv_labels,times=eval_times, metric="brier_score")["perf"].values
                
            ## numerical feature
            else:
                feats_perm_df = numerical_perturbation()
                bs_perf_perturbated = performance.evaluate(explainer, feats_perm_df, surv_labels,times=eval_times, metric="brier_score")["perf"].values
                
    
            
            ### ratio/difference after/before perturbation
            if type == "ratio":
                importance_ratio = bs_perf / bs_perf_perturbated
                importance_ratio_data = np.stack(([feat_name_org] * n_eval_times, eval_times, importance_ratio)).T
                additional_ratio = pd.DataFrame(data=importance_ratio_data, columns=feat_importance_df_cols)
                feat_importance_df = pd.concat([feat_importance_df, additional_ratio], ignore_index=False)

    feat_importance_df[["times", "perf"]] = feat_importance_df[["times", "perf"]].apply(pd.to_numeric)

    return feat_importance_df


def numerical_perturbation():
    
def categorical_perturbation():
    
def unique_categorical_perturbation():
    
    

def permutation_feature_importance(explainer, feats, surv_labels, eval_times=None,
                                   n_perm = 10, loss="brier_score", type="ratio"):
	"""
	Compute permutation feature importance (PFI)


	Parameters
	----------
	explainer : `class`
		A Python class used to explain the survival model

	Returns
	-------
	PFI_df : `np.ndarray`, shape=()
		Returns the PFI value of selected features
	"""

	if loss == "brier_score":
		bs_perf = performance.evaluate(explainer, feats, surv_labels,
		                               times=eval_times, metric="brier_score")["perf"].values

		feats_name = feats.columns.values.tolist()
		feat_importance_df_cols = ["feat", "times", "perf"]
		feat_importance_df = pd.DataFrame(columns=feat_importance_df_cols)
		n_eval_times  = len(eval_times)

		feats_name_org = explainer.numeric_feats + explainer.cate_feats
		feats_name_ext = explainer.numeric_feats
		for cate_feat_name in explainer.cate_feats:
			cate_feat_name_list = []
			for feat_name in feats_name:
				if cate_feat_name in feat_name:
					cate_feat_name_list.append(feat_name)
			if len(cate_feat_name_list):
				feats_name_ext.append(cate_feat_name_list)

		for i in range(len(feats_name_ext)):
			feat_name = feats_name_ext[i]
			feat_name_org = feats_name_org[i]
			bs_perf_perm = np.zeros(n_eval_times)
			for k in range(n_perm):
				feat_perm = feats.copy(deep=True)[feat_name].values
				np.random.shuffle(feat_perm)
				feats_perm_df = feats.copy(deep=True)
				feats_perm_df[feat_name] = feat_perm
				tmp = performance.evaluate(explainer, feats_perm_df, surv_labels,
				                           times=eval_times, metric="brier_score")["perf"].values
				bs_perf_perm += (1 / n_perm) * tmp

			if type == "ratio":
				importance_ratio = bs_perf / bs_perf_perm
				importance_ratio_data = np.stack(([feat_name_org] * n_eval_times, eval_times, importance_ratio)).T
				additional_ratio = pd.DataFrame(data=importance_ratio_data, columns=feat_importance_df_cols)
				feat_importance_df = pd.concat([feat_importance_df, additional_ratio], ignore_index=False)

	feat_importance_df[["times", "perf"]] = feat_importance_df[["times", "perf"]].apply(pd.to_numeric)

	return feat_importance_df