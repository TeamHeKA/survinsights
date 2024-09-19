# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:34:49 2024

@author: ducro
"""

##Example taken from sksurv https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html

from sksurv.datasets import load_veterans_lung_cancer

data_x, data_y = load_veterans_lung_cancer()
data_y


import pandas as pd

pd.DataFrame.from_records(data_y[[11, 5, 32, 13, 23]], index=range(1, 6))


%matplotlib inline
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

time, survival_prob, conf_int = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"],conf_type="log-log")


plt.step(time, survival_prob, where="post")
plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
plt.ylim(0, 1)
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()

data_x["Treatment"].value_counts()

for treatment_type in ("standard", "test"):
    mask_treat = data_x["Treatment"] == treatment_type
    time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(data_y["Status"][mask_treat],data_y["Survival_in_days"][mask_treat],conf_type="log-log",)

    plt.step(time_treatment, survival_prob_treatment, where="post", label=f"Treatment = {treatment_type}")
    plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")

plt.ylim(0, 1)
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()


for value in data_x["Celltype"].unique():
    mask = data_x["Celltype"] == value
    time_cell, survival_prob_cell, conf_int = kaplan_meier_estimator(
        data_y["Status"][mask], data_y["Survival_in_days"][mask], conf_type="log-log"
    )
    plt.step(time_cell, survival_prob_cell, where="post", label=f"{value} (n = {mask.sum()})")
    plt.fill_between(time_cell, conf_int[0], conf_int[1], alpha=0.25, step="post")

plt.ylim(0, 1)
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()



from sksurv.preprocessing import OneHotEncoder

data_x_numeric = OneHotEncoder().fit_transform(data_x)
data_x_numeric.head()


from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis

set_config(display="text")  # displays text representation of estimators

model = CoxPHSurvivalAnalysis()
model.fit(data_x_numeric, data_y)


model.predict_survival_function(np.array([65, 0, 0, 1, 60, 1, 0, 1]).reshape(1, -1))


results = model.predict_survival_function(np.array([[65, 0, 0, 1, 60, 1, 0, 1], [69, 0, 0, 0, 60, 1, 0, 1]]))


newdata = [[65, 0, 0, 1, 60, 1, 0, 1], [69, 0, 0, 0, 60, 1, 0, 1]]

np.array(newdata).shape
np.array(model.predict_survival_function(np.array(newdata).reshape(1, -1))[0](times))

model.predict_survival_function(newdata)


np.array(data_x_numeric)


explainer=SurvExplainer(model=model,data=data_x_numeric,y= data_y, from_package="Sksurv")


#explainer.times
explainer.model
explainer.data.shape

times= [i/100 for i in range(0,50000)]
newdata=[-0.7185573 ,-0.73641497, 2.2446876 , 1.0520973 , -1.1191831 ,0. ,  1. ,  1. ]

newdata=np.array(data_x_numeric)[1]
explainer.predict_survival_function(newdata=newdata,times=times)

newdata=np.array(data_x_numeric)[1:2]
explainer.predict_survival_function(newdata=newdata,times=times)

newdata=np.array(data_x_numeric)[1:4]
explainer.predict_survival_function(newdata=newdata,times=times)

explainer.predict_cumulative_hazard_function(newdata=newdata,times=times)




