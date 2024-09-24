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

	def __init__(self, model, data, label, times = None,
	             time_generation="quantile", sf = None, chf = None):

		self.model = model
		# TODO: Check the availability of data, label
		self.data = data
		self.label = label

		self.X = self.data.values

		if sf is not None:
			self.sf = sf
		else:
			self.sf = model.predict_survival_function

		if chf is not None:
			self.chf = chf
		else:
			self.chf = model.predict_cumulative_hazard_function

		if times is None:
			survival_times = label[:, 0]

			if time_generation == "quantile":
				qt_list = np.arange(0.05, 0.95, 0.05)
				self.times = np.quantile(survival_times, qt_list)

			elif time_generation == "uniform":
				self.times = np.linspace(min(survival_times), max(survival_times), 50)

			else:
				self.times = np.unique(survival_times)[::10]
