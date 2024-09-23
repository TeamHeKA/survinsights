# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:50:18 2024

@author: ducro
"""

import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

import pycox
from pycox.datasets import metabric
from pycox.models import LogisticHazard
# from pycox.models import PMF
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv


df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

df_train.head()




cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)


x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')



num_durations = 10

labtrans = LogisticHazard.label_transform(num_durations)
# labtrans = PMF.label_transform(num_durations)
#labtrans = DeepHitSingle.label_transform(num_durations)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

type(labtrans)
labtrans.cuts
y_train
labtrans.cuts[y_train[0]]




in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)



model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)



batch_size = 256
epochs = 100
callbacks = [tt.cb.EarlyStopping()]

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)


_ = log.plot()
plt.show()

log.to_pandas().val_loss.min()
model.score_in_batches(val)



surv = model.interpolate(100).predict_surv_df(x_test)

model.interpolate(10)


surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()


##### EXPLAINER for the defined model (pycox Logistic Hazard model with interpolation and METABRIC dataset)

explainer=SurvExplainer(model=model.interpolate(100),data=x_train,y=y_train, from_package="Pycox")

##### Quick test for predict_survival_function and predict_cumulative_hazard_function
explainer.model
explainer.data.shape

times= [i/100 for i in range(0,50000)]
newdata=[-0.7185573 ,-0.73641497, 2.2446876 , 1.0520973 , -1.1191831 ,0. ,  1. ,  1. ,  0.]

explainer.predict_survival_function(newdata=x_test[1,:],times=times)

explainer.predict_survival_function(newdata=x_test[1:3,:],times=times)

explainer.predict_survival_function(newdata=x_test[1:2,:],times=times)

explainer.predict_cumulative_hazard_function(newdata=x_test[1:2,:],times=times)

