import numpy as np
from scipy.interpolate import interp1d

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
    return -np.log(predict_survival_function_pycox(model,newdata))  