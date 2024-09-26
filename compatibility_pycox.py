# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:14:28 2024

@author: ducro
"""

from scipy.interpolate import interp1d
import numpy as np

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
    return -np.log(predict_survival_function_pycox(model,newdata))  