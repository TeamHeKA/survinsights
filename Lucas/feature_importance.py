# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:03:27 2024

@author: ducro
"""

from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score
from sksurv.preprocessing import OneHotEncoder
from random import shuffle


X, y = load_gbsg2()
X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
Xt = OneHotEncoder().fit_transform(X)

est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

survs = est.predict_survival_function(Xt)
preds = [fn(1825) for fn in survs]

times, score = brier_score(y, y, preds, 1825)
print(score)



survs = model.predict_survival_function(data_x_numeric)

explainer=SurvExplainer(model=model,data=data_x_numeric,y= data_y, from_package="Sksurv")

t= [i/10 for i in range(10,50)]
preds = explainer.predict_survival_function(newdata = np.array(data_x_numeric),times = t)


temp_times, temp_score = brier_score(data_y, data_y, preds, t)
times, score = brier_score(y, y, preds, times)

brier_score(data_y, data_y, preds, t)[1]


def feature_importance(variable,explainer,data,y,times,nb_resampling=100):
    preds = explainer.predict_survival_function(np.array(data),times)
    model_BS = brier_score(y, y, preds, times)
    
    model_BS_resample = []
    
    for i in range(0,nb_resampling):
        data_resample = np.array(data)
        shuffle(data_resample[:,variable])
        preds_resample = explainer.predict_survival_function(np.array(data_resample),times)
        model_BS_resample.append(np.array(brier_score(y, y, preds_resample, times)[1]))
        
    #return np.array(np.sum(np.array(model_BS_resample),axis=0)/nb_resampling)-np.array(model_BS[1])
    return np.array(np.sum(np.array(model_BS_resample),axis=0)/nb_resampling)-np.array(model_BS[1])
        
        
        
        

variable2 = feature_importance(2, explainer, data_x_numeric, data_y, t) 
variable3 = feature_importance(3, explainer, data_x_numeric, data_y, t)
variable4 = feature_importance(4, explainer, data_x_numeric, data_y, t)
variable5 = feature_importance(5, explainer, data_x_numeric, data_y, t) 
variable6 = feature_importance(6, explainer, data_x_numeric, data_y, t) 
variable7 = feature_importance(7, explainer, data_x_numeric, data_y, t)
        
plt.plot(t, variable2, marker='o', linestyle='-', color='b', label='Ligne de points')
plt.plot(t, variable3, marker='o', linestyle='-', color='r', label='Ligne de points')
plt.plot(t, variable4, marker='o', linestyle='-', color='g', label='Ligne de points')
plt.plot(t, variable5, marker='o', linestyle='-', color='brown', label='Ligne de points')
plt.plot(t, variable6, marker='o', linestyle='-', color='cyan', label='Ligne de points')
plt.plot(t, variable7, marker='o', linestyle='-', color='aqua', label='Ligne de points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Feature Importance')
plt.show()
        
A = []    
A.append([3,8,0])   
A.append([5,6,1])        
        
test = np.array(data_x_numeric)

random.shuffle(test[:,3])
a = test[:,3]
shuffle(a)
print(x)


np.array([1,8,0])-np.array([0,4,5])



preds = explainer.predict_survival_function(np.array(data_x_numeric),t)
model_BS = brier_score(data_y, data_y, preds, t)

model_BS_resample = []

for i in range(0,10):
    data_resample = np.array(data_x_numeric)
    shuffle(data_resample[:,1])
    preds_resample = explainer.predict_survival_function(np.array(data_resample),t)
    model_BS_resample.append(brier_score(data_y, data_y, preds_resample, times)[1])
    
results = np.array(np.sum(np.array(model_BS_resample),axis=0)/10)-np.array(model_BS[1])
