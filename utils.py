# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:36:40 2024

@author: ducro
"""


def Brier_Score(explainer, data, y, t):
    prob_surv=explainer.predict_survival_function(newdata = data, times = t)
    
    
    
