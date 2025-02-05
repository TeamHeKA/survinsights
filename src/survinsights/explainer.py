import numpy as np


class explainer:
    """
        A class to define the explaination model

        Parameters
    ----------
    model
        A survival model to be explained

    data :  `pd.DataFrame`, shape=(n_samples, n_features)
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

    def __init__(
        self,
        model,
        features_df,
        survival_labels,
        times=None,
        time_generation="quantile",
        survival_fucntion=None,
        cummulative_hazard_function=None,
        encoders=None,
    ):
        self.model = model
        # TODO: Check the availability of data, label
        self.features_df = features_df
        self.survival_labels = survival_labels

        self.features_value = self.features_df.values

        if survival_fucntion is not None:
            self.sf = survival_fucntion
        elif "sksurv" in model.__module__:
            self.sf = model.predict_survival_function
        elif "pycox" in model.__module__:
            self.sf = model.predict_surv_df
        else:
            msg = "Unsupported model"
            raise ValueError(msg)

        if cummulative_hazard_function is not None:
            self.chf = cummulative_hazard_function
        elif "sksurv" in model.__module__:
            self.chf = model.predict_cumulative_hazard_function
        elif "pycox" in model.__module__:
            self.chf = model.predict_cumulative_hazards
        else:
            msg = "Unsupported model"
            raise ValueError(msg)

        if times is None:
            surv_times, surv_indx = survival_labels[:, 0], survival_labels[:, 1]

            if time_generation == "quantile":
                qt_list = np.arange(0.05, 0.95, 0.05)
                self.times = np.quantile(surv_times[surv_indx == 1], qt_list)

            elif time_generation == "uniform":
                self.times = np.linspace(
                    np.quantile(surv_times[surv_indx == 1], 0.1),
                    np.quantile(surv_times[surv_indx == 1], 0.9),
                    50,
                )

            else:
                self.times = np.unique(surv_times[surv_indx == 1])[::10]
        else:
            self.times = times

        self.encoders = encoders
        if encoders:
            self.cate_feat_names = list(encoders.keys())
            self.numeric_feat_names = [
                feat_name
                for feat_name in features_df.columns.values
                if not np.array([cate_feat in feat_name for cate_feat in self.cate_feat_names]).any()
            ]
            self.feat_names = self.numeric_feat_names + self.cate_feat_names
        else:
            self.cate_feat_names = None
            self.numeric_feat_names = list(features_df.columns.values)
            self.feat_names = self.numeric_feat_names
