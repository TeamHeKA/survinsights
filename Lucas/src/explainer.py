import numpy as np

class explainer():
    """
	A class to define the explaination model

	Parameters
    ----------
    model
        A survival model to be explained

    data :  `np.ndarray`, shape=(n_samples, n_features)
        Covariates of new observations need to be explained

    label :  `np.ndarray`, shape=(n_samples, 2), default = None
		Survival label of new observations

    time_generation : `str`, default = "quantile"
        Method used to generate times

    sf :
        Method to predict survival function

    chf :
        Method to predict cumulative hazard function
	"""

    def __init__(self, model, data, label, times = None, time_generation="quantile",survival_fucntion  = None, cummulative_hazard_function = None, encoders=None):
        self.model = model
        # TODO: Check the availability of data, label
        self.data = data
        self.label = label
        self.X = self.data.values
        self.times = times

        if survival_fucntion is not None:
            self.sf = survival_fucntion
        elif "sksurv" in model.__module__:
            self.sf = model.predict_survival_function
        elif "pycox" in model.__module__:
            self.sf = model.predict_surv_df
        else:
            raise ValueError("Unsupported model")

        if cummulative_hazard_function is not None:
            self.chf = cummulative_hazard_function
        elif "sksurv" in model.__module__:
            self.chf = model.predict_cumulative_hazard_function
        elif "pycox" in model.__module__:
            self.chf = model.predict_cumulative_hazards
        else:
            raise ValueError("Unsupported model")

        if times is None:
            survival_times = label[:, 0]

            if time_generation == "quantile":
                qt_list = np.arange(0.05, 0.95, 0.05)
                self.times = np.quantile(survival_times, qt_list)

            elif time_generation == "uniform":
                self.times = np.linspace(min(survival_times), max(survival_times), 50)

            else:
                self.times = np.unique(survival_times)[::10]

        self.encoders = encoders
        if encoders is not None:
            self.cate_feats = list(encoders.keys())
            numeric_feats = []
            for feat in data.columns.values:
                if not np.array([cate_feat in feat for cate_feat in self.cate_feats]).any():
                    numeric_feats.append(feat)
            self.numeric_feats = numeric_feats
        else:
            self.cate_feats = None
            self.numeric_feats = list(data.columns.values)
