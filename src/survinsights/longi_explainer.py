import numpy as np

class explainer():
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

	def __init__(self, model, model_name, paths, survival_labels, static_feats=None):

		self.model = model
		self.model_name = model_name
		# TODO: Check the availability of data, label
		self.paths = paths
		self.survival_labels = survival_labels
		self.sf = model.predict_survival
		self.risk = model.predict_hazard
		self.static_feats = static_feats