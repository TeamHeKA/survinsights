# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 08:55:06 2024

@author: ducro
"""

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

#' @param model object - a survival model to be explained
#' @param data data.frame - data which will be used to calculate the explanations. If not provided, then it will be extracted from the model if possible. It should not contain the target columns. NOTE: If the target variable is present in the `data` some functionality breaks.
#' @param y `survival::Surv` object containing event/censoring times and statuses corresponding to `data`
#' @param predict_function  function taking 2 arguments - `model` and `newdata` and returning a single number for each observation - risk score. Observations with higher score are more likely to observe the event sooner.
#' @param predict_function_target_column unused, left for compatibility with DALEX
#' @param residual_function unused, left for compatibility with DALEX
#' @param weights unused, left for compatibility with DALEX
#' @param ... additional arguments, passed to `DALEX::explain()`
#' @param label character - the name of the model. Used to differentiate on visualizations with multiple explainers. By default it's extracted from the 'class' attribute of the model if possible.
#' @param verbose logical, if TRUE (default) then diagnostic messages will be printed
#' @param colorize logical, if TRUE (default) then WARNINGS, ERRORS and NOTES are colorized. Will work only in the R console. By default it is FALSE while knitting and TRUE otherwise.
#' @param model_info a named list (`package`, `version`, `type`) containing information about model. If `NULL`, `survex` will seek for information on its own.
#' @param type type of a model, by default `"survival"`
#'
#' @param times numeric, a vector of times at which the survival function and cumulative hazard function should be evaluated for calculations
#' @param times_generation either `"survival_quantiles"`, `"uniform"` or `"quantiles"`. Sets the way of generating the vector of times based on times provided in the `y` parameter. If `"survival_quantiles"` the vector contains unique time points out of 50 uniformly distributed survival quantiles based on the Kaplan-Meier estimator, and additional time point being the median survival time (if possible); if `"uniform"` the vector contains 50 equally spaced time points between the minimum and maximum observed times; if `"quantiles"` the vector contains unique time points out of 50 time points between 0th and 98th percentiles of observed times. Ignored if `times` is not `NULL`.
#' @param predict_survival_function function taking 3 arguments `model`, `newdata` and `times`, and returning a matrix whose each row is a survival function evaluated at `times` for one observation from `newdata`
#' @param predict_cumulative_hazard_function function taking 3 arguments `model`, `newdata` and `times`, and returning a matrix whose each row is a cumulative hazard function evaluated at `times` for one observation from `newdata`
#'
#' @return It is a list containing the following elements:
#'
#' * `model` - the explained model.
#' * `data` - the dataset used for training.
#' * `y` - response for observations from `data`.
#' * `residuals` - calculated residuals.
#' * `predict_function` - function that may be used for model predictions, shall return a single numerical value for each observation.
#' * `residual_function` - function that returns residuals, shall return a single numerical value for each observation.
#' * `class` - class/classes of a model.
#' * `label` - label of explainer.
#' * `model_info` - named list containing basic information about model, like package, version of package and type.
#' * `times` - a vector of times, that are used for evaluation of survival function and cumulative hazard function by default
#' * `predict_survival_function` - function that is used for model predictions in the form of survival function
#' * `predict_cumulative_hazard_function` - function that is used for model predictions in the form of cumulative hazard function
#'


class SurvExplainer:
    def __init__(self, model, data=None, y=None, from_package=None):
    
        #, times=None,
        #             predict_survival_function=None,predict_cumulative_hazard_function=None):
        
        
        # Initialize attributes
        self.model = model
        self.data = data
        #self.y = y
        self.y=y
        self.from_package=from_package
        #self.times = times
        #self.censor_flag = censor_flag
        
        #self.predict_cumulative_hazard_function=predict_cumulative_hazard_function
        #self.y_hat = y_hat
        #self.residual_function = residual_function
        #self.residuals = residuals
        #self.weights = weights
        #self.label = label
        #self.model_class = model_class
        #self.model_type = model_type
        #self.model_info = _model_info
        
        #if (data == None):
        #    self.data = model.data
        #else:
        #    self.data = data
            
        #if (predict_survival_fun == None):
        #    self.predict_survival_fun = self.predict_survival_fun
        #else:
        #    self.data = data
    def predict_survival_function(self, newdata, times):
        if self.from_package == "Pycox":
            return self.predict_survival_function_pycox(newdata, times)
        elif self.from_package == "Sksurv":
            return self.predict_survival_function_sksurv(newdata, times)
        else:
            return None
        
    
    def predict_survival_function_sksurv(self, newdata, times):
        

        if len(np.array(newdata).shape)==1:
            return np.array(self.model.predict_survival_function(np.array(newdata).reshape(1, -1))[0](times))

            
        else:
            surv = []
            for i in range(np.array(newdata).shape[0]):
                line = np.array(self.model.predict_survival_function(np.array(newdata[i]).reshape(1, -1))[0](times))
                surv.append(line)
            surv = np.array(surv)

            return surv
    


    def predict_survival_function_pycox(self, newdata, times):

        if len(newdata.shape)==1:
            surv=self.model.predict_surv_df(np.array([newdata]))
            t=np.array([surv.index])
            surv_prob=np.array([surv])
                
            surv_func = interp1d(np.insert(t,0,0), np.insert(surv_prob,0,1), kind='previous', fill_value="extrapolate")
                
            return np.array(surv_func(times))
            
        elif len(newdata.shape)==2 and newdata.shape[0]==1:
            surv=self.model.predict_surv_df(newdata)
            t=np.array([surv.index])
            surv_prob=np.array([surv])
                
            surv_func = interp1d(np.insert(t,0,0), np.insert(surv_prob,0,1), kind='previous', fill_value="extrapolate")
                
            return np.array(surv_func(times))
            
        else:
                
            surv=self.model.predict_surv_df(newdata)
            t=np.array([surv.index])
                
            surv_prob=np.array(surv)
                
            results=np.zeros((surv_prob.shape[1],len(times)))
                
            for i in range(0,surv_prob.shape[1],1):
                    
                surv_prob_ind=surv_prob[:,i]
                surv_func = interp1d(np.insert(t,0,0), np.insert(surv_prob_ind,0,1), kind='previous', fill_value="extrapolate")
                    
                results[i,:]=surv_func(times)
                
            return np.array(results)


    def predict_cumulative_hazard_function(self, newdata, times):

        return -np.log(self.predict_survival_function(newdata, times))





   