

'''
First implemented: 01/25/2018
  > For survival analysis on longitudinal dataset
By CHANGHEE LEE

Modifcation List:
    - (02/13/2018) C-index, B-score evaluation added (using pred_time and eval_time)
    - (02/13/2018) Valdiation Added (frist version is based on the mean of C-index w/ p_time=1 and e_time=3)
    - (02/14/2018) Prediction modified (divided by the denominator)
    - (02/15/2018) Cystic-Fibrosis Added
    - (02/21/2018) Comorbidity indes added (specific inidces can be selected among multiple features)
    - (02/22/2018) Burn-in training for RNN is added
    - (02/28/2018) Boosting training Set is added (N longitudinal measurements --> N samples with 1~N longitudinal measurements)
'''
_EPSILON = 1e-08


import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils

from class_DeepLongitudinal import Model_Longitudinal_Attention
from utils_eval import c_index, brier_score


##### USER-DEFINED FUNCTIONS
def log(x): 
    return tf.log(x + 1e-8)

def div(x, y):
    return tf.div(x, (y + 1e-8))


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

    return mask

def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss 
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask

def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask


def f_get_minibatch(mb_size, x, x_org, x_mi, label, time, mask1, mask2, mask3, mask4, mask5):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb     = x[idx, :, :].astype(float)
    x_org_mb = x_org[idx, :, :].astype(float)
    x_mi_mb  = x_mi[idx, :, :].astype(float)
    k_mb     = label[idx, :].astype(float) # censoring(0)/event(1,2,..) label
    t_mb     = time[idx, :].astype(float)
    m1_mb    = mask1[idx, :, :].astype(float) #rnn_mask
    m2_mb    = mask2[idx, :, :].astype(float) #rnn_mask
    m3_mb    = mask3[idx, :, :].astype(float) #fc_mask
    m4_mb    = mask4[idx, :, :].astype(float) #fc_mask
    m5_mb    = mask5[idx, :].astype(float) #fc_mask
    return x_mb, x_org_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb, m5_mb

###MODIFY
def f_get_prediction_v5(model, data, data_org, data_mi, time, label, mask2, pred_horizon):
    """
    TF2 version: removed sess; uses model.predict(...) directly.
    """
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))
    new_time    = np.zeros(np.shape(time))
    new_label   = np.zeros(np.shape(label))

    new_mask2   = np.zeros(np.shape(mask2))
    last_meas   = np.zeros([np.shape(data)[0], 1])

    for i in range(np.shape(data)[0]):
        if np.max(data_org[i, :, 1]) <= pred_horizon:
            new_data[i, :, :]    = data[i, :, :]
            new_data_mi[i, :, :] = data_mi[i, :, :]
            new_mask2[i, :, :]   = mask2[i, :, :]
            last_meas[i, 0]      = np.max(data_org[i, :, 1])
        elif np.min(data_org[i, data_org[i, :, 1] > 0, 1]) <= pred_horizon:
            for t in range(np.shape(data)[1]):
                if data_org[i, t, 1] <= pred_horizon:
                    new_data[i, t, :]    = data[i, t, :]
                    new_data_mi[i, t, :] = data_mi[i, t, :]
                    last_meas[i, 0]      = data_org[i, t, 1]
                else:
                    new_mask2[i, t - 1, :] = 1
                    break

    keep = np.where(np.sum(np.sum(new_data, axis=2), axis=1) != 0)[0]
    last_meas   = last_meas[keep, :]
    new_time    = time[keep, :]
    new_label   = label[keep, :]
    new_mask2   = new_mask2[keep, :, :]
    new_data_mi = new_data_mi[keep, :, :]
    new_data    = new_data[keep, :, :]

    # IMPORTANT: your TF2 model.predict signature should be (x, x_mi, keep_prob=1.0)
    pred = model.predict(new_data, new_data_mi, keep_prob=1.0)
    return pred, new_time, new_label, last_meas


def get_valid_performance(in_parser, out_itr, MAX_VALUE=-99, OUT_ITERATION=5, seed=1234):
    """
    TF2 rewrite of your TF1 get_valid_performance.

    Key changes:
    - removed reset_default_graph/session/config
    - replaced tf.train.Saver with tf.train.Checkpoint + CheckpointManager
    - everything else (data split, minibatch loops, early stopping logic) kept the same
    """

    # --- same feature selection logic as your TF1 code ---
    if out_itr == 0:
        selected_feat = [0, 1, 2, 5, 6, 9, 13, 24, 29, 34, 37, 39, 45, 47, 56, 60, 64, 66, 84, 86, 89, 90]
        x_dim, x_dim_cont = len(selected_feat), 5
        x_dim_bin = x_dim - 1 - x_dim_cont
    elif out_itr == 1:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 13, 24, 37, 39, 43, 44, 45, 56, 60, 62, 64, 67, 68, 70, 73, 84, 86, 90]
        x_dim, x_dim_cont = len(selected_feat), 6
        x_dim_bin = x_dim - 1 - x_dim_cont
    elif out_itr == 2:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 24, 31, 37, 56, 60, 62, 64, 67, 68, 73, 84]
        x_dim, x_dim_cont = len(selected_feat), 6
        x_dim_bin = x_dim - 1 - x_dim_cont
    elif out_itr == 3:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 24, 36, 37, 39, 45, 47, 56, 59, 60, 61, 64, 66, 73, 84]
        x_dim, x_dim_cont = len(selected_feat), 6
        x_dim_bin = x_dim - 1 - x_dim_cont
    elif out_itr == 4:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 19, 24, 31, 37, 45, 56, 59, 62, 64, 66, 67, 73, 84]
        x_dim, x_dim_cont = len(selected_feat), 6
        x_dim_bin = x_dim - 1 - x_dim_cont
    else:
        raise ValueError("out_itr out of expected range 0..4")

    # These are globals in your script; keep as-is if your module defines them:
    # data, data_org, data_mi, time, label, mask1, mask2, mask3, mask4, mask5
    tmp_data     = data[:, :, selected_feat]
    tmp_data_org = data_org[:, :, selected_feat]
    tmp_data_mi  = data_mi[:, :, selected_feat]
    tmp_mask1    = mask1[:, :, selected_feat]
    tmp_mask2    = mask2[:, :, selected_feat]

    # --- hyperparameters ---
    mb_size           = in_parser['mb_size']
    iteration_burn_in = in_parser['iteration_burn_in']
    iteration         = in_parser['iteration']
    keep_prob         = in_parser['keep_prob']
    lr_train          = in_parser['lr_train']

    alpha = in_parser['alpha']
    beta  = in_parser['beta']
    gamma = in_parser['gamma']

    # TF2 Xavier/Glorot initializer replacement
    initial_W = tf.keras.initializers.GlorotUniform()

    # --- build input/network dicts (same keys as TF1) ---
    _, num_Event, num_Category = np.shape(mask3)
    max_length = np.shape(tmp_data)[1]

    input_dims = {
        'x_dim': x_dim,
        'x_dim_cont': x_dim_cont,
        'x_dim_bin': x_dim_bin,
        'num_Event': num_Event,
        'num_Category': num_Category,
        'max_length': max_length,
    }

    network_settings = {
        'h_dim_RNN': in_parser['h_dim_RNN'],
        'h_dim_FC': in_parser['h_dim_FC'],
        'num_layers_RNN': in_parser['num_layers_RNN'],
        'num_layers_ATT': in_parser['num_layers_ATT'],
        'num_layers_CS': in_parser['num_layers_CS'],
        'RNN_type': in_parser['RNN_type'],
        'BiRNN': in_parser['BiRNN'],
        'FC_active_fn': ACTIVATION_FN[in_parser['FC_active_fn']],
        'RNN_active_fn': ACTIVATION_FN[in_parser['RNN_active_fn']],
        'initial_W': initial_W,
        # If your TF2 model expects these:
        'reg_W': in_parser.get('reg_W', 0.0),
        'reg_W_out': in_parser.get('reg_W_out', 0.0),
    }

    file_path = in_parser['out_path'] + '/valid/itr_' + str(out_itr)
    file_path_final = in_parser['out_path'] + '/itr_' + str(out_itr)

    os.makedirs(file_path + '/results/', exist_ok=True)
    os.makedirs(file_path + '/models/', exist_ok=True)
    os.makedirs(file_path_final + '/models/', exist_ok=True)

    pred_time = [30, 40, 50]
    eval_time = [1, 3, 5, 10]

    print('ITR: ' + str(out_itr + 1) + ' DATA MODE: ' + data_mode +
          ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')')

    # --- Create TF2 model (sess arg kept for compatibility, pass None) ---
    model = Model_Longitudinal_Attention(None, "FHT_Landmarking", input_dims, network_settings)

    # --- TF2 checkpointing (replaces Saver) ---
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, directory=file_path + '/models/', max_to_keep=1)
    manager_final = tf.train.CheckpointManager(ckpt, directory=file_path_final + '/models/', max_to_keep=1)

    # --- Split train/val/test (same as TF1) ---
    (tr_data, te_data, tr_data_org, te_data_org, tr_data_mi, te_data_mi, tr_time, te_time,
     tr_label, te_label, tr_mask1, te_mask1, tr_mask2, te_mask2, tr_mask3, te_mask3,
     tr_mask4, te_mask4, tr_mask5, te_mask5) = train_test_split(
        tmp_data, tmp_data_org, tmp_data_mi, time, label, tmp_mask1, tmp_mask2, mask3, mask4, mask5,
        test_size=0.2, random_state=seed + out_itr
    )

    (tr_data, va_data, tr_data_org, va_data_org, tr_data_mi, va_data_mi, tr_time, va_time,
     tr_label, va_label, tr_mask1, va_mask1, tr_mask2, va_mask2, tr_mask3, va_mask3,
     tr_mask4, va_mask4, tr_mask5, va_mask5) = train_test_split(
        tr_data, tr_data_org, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5,
        test_size=0.2, random_state=seed
    )

    if boost_mode == 'ON':
        tr_data, tr_data_org, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5 = \
            f_get_boosted_trainset(tr_data, tr_data_org, tr_data_mi, tr_time, tr_label,
                                   tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5)

    # --- Burn-in training ---
    if burn_in_mode == 'ON':
        print("BURN-IN TRAINING ...")
        for itr in range(iteration_burn_in):
            x_mb, x_org_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb, m5_mb = \
                f_get_minibatch(mb_size, tr_data, tr_data_org, tr_data_mi, tr_label, tr_time,
                                tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5)

            # Your TF2 model.train_burn_in signature may differ; adapt if needed.
            # In the TF2 class I gave earlier it was: train_burn_in(DATA, MISSING, keep_prob, lr_train)
            DATA = (x_mb, x_org_mb, k_mb, t_mb)
            MISSING = (x_mi_mb,)

            _, loss_curr = model.train_burn_in(DATA, MISSING, keep_prob, lr_train)

            if (itr + 1) % 1000 == 0:
                print('|| Epoch: ' + str('%04d' % (itr + 1)) +
                      ' | Loss: ' + colored(str('%.4f' % (loss_curr)), 'green', attrs=['bold']))

    max_valid = -99
    stop_flag = 0

    # --- Main training ---
    print("MAIN TRAINING ...")
    for itr in range(iteration):
        if stop_flag > 5:
            break

        x_mb, x_org_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb, m5_mb = \
            f_get_minibatch(mb_size, tr_data, tr_data_org, tr_data_mi, tr_label, tr_time,
                            tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5)

        DATA = (x_mb, x_org_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb, m3_mb, m4_mb, m5_mb)
        MISSING = (x_mi_mb,)
        PARAMETERS = (alpha, beta, gamma)

        _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)

        if (itr + 1) % 1000 == 0:
            print('|| Epoch: ' + str('%04d' % (itr + 1)) +
                  ' | Loss: ' + colored(str('%.4f' % (loss_curr)), 'yellow', attrs=['bold']))

        # --- Validation every 1000 iters (same logic) ---
        if (itr + 1) % 1000 == 0 and valid_mode == 'ON':
            for p, p_time in enumerate(pred_time):
                pred_horizon = int(p_time / time_interval)

                pred, tmp_time, tmp_label, _ = f_get_prediction_v5(
                    model, va_data, va_data_org, va_data_mi, va_time, va_label, va_mask2, pred_horizon
                )

                val_result1 = np.zeros([num_Event, len(eval_time)])

                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time / time_interval) + pred_horizon
                    if eval_horizon >= num_Category:
                        print('ERROR: evaluation horizon is out of range')
                        val_result1[:, t] = 0
                    else:
                        risk = np.sum(pred[:, :, pred_horizon:(eval_horizon + 1)], axis=2)
                        risk = risk / (np.sum(np.sum(pred[:, :, pred_horizon:], axis=2), axis=1, keepdims=True) + _EPSILON)
                        for k in range(num_Event):
                            val_result1[k, t] = c_index(
                                risk[:, k], tmp_time, (tmp_label[:, 0] == k + 1).astype(int), eval_horizon
                            )

                val_final1 = val_result1 if p == 0 else np.append(val_final1, val_result1, axis=0)

            tmp_valid = np.mean(val_final1)

            if tmp_valid > max_valid:
                stop_flag = 0
                max_valid = tmp_valid

                # Save "best so far"
                manager.save()
                print('updated.... average c-index = ' + str('%.4f' % (tmp_valid)))

                if max_valid > MAX_VALUE:
                    manager_final.save()
            else:
                stop_flag += 1

    return max_valid
