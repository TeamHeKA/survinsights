# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:02:01 2024

@author: ducro
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv


np.random.seed(1234)
_ = torch.manual_seed(123)

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


get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val


in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,dropout, output_bias=output_bias)


model = CoxPH(net, tt.optim.Adam)

batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
plt.show()



lrfinder.get_best_lr()


model.optimizer.set_lr(0.01)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True



log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val, val_batch_size=batch_size)


_ = log.plot()
plt.show()


model.partial_log_likelihood(*val).mean()


_ = model.compute_baseline_hazards()


surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()


##### EXPLAINER for the defined model (pycox CoxPH model and METABRIC dataset)

explainer=SurvExplainer(model=model,data=x_train,y=y_train)

##### Quick test for predict_survival_function and predict_cumulative_hazard_function

explainer.model
explainer.data.shape

times= [i/100 for i in range(0,50000)]
newdata=[-0.7185573 ,-0.73641497, 2.2446876 , 1.0520973 , -1.1191831 ,0. ,  1. ,  1. ,  0.]

explainer.predict_survival_function(newdata=x_test[1,:],times=times)

explainer.predict_survival_function(newdata=x_test[1:3,:],times=times)

explainer.predict_survival_function(newdata=x_test[1:2,:],times=times)

explainer.predict_cumulative_hazard_function(newdata=x_test[1:2,:],times=times)
