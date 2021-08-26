from typing import Dict
from utilities import *
import logging
import traceback
import os, sys
from time import time
import numpy as np
import pandas as pd

# arima forecast
from pmdarima import arima
from pmdarima import pipeline
from pmdarima import preprocessing

# holt-winter's forecast
from statsmodels.tsa.api import ExponentialSmoothing

# from modeling import Modeling
# from dataset import Dataset

class BaseModel(object):

    def __init__(self):
        self.logger = logging.getLogger('planiqum_forecast_application.' + __name__)
        model = None
        self.y_pred = None
        self.modeling = None


    def fit(self):
        pass

    def predict(self):
        pass

    def estimate(self):
        pass

    def smape_score(self):
        actuals = self.modeling.y_test
        forecast = self.y_pred
        return (100/len(actuals) * np.sum(np.abs((forecast - actuals)) / ((np.abs(actuals) + np.abs(forecast)) / 2)))

    def rmse_score(self):
        actuals = self.modeling.y_test
        forecast = self.y_pred
        return (np.sum(np.power((forecast - actuals), 2)) / len(actuals))**(1/2)



class ArimaModel(BaseModel):

    def __init__(self, modeling, name, params):
        super().__init__()
        self.modeling = modeling
        self.model_name = name
        self.verify_params(params)
        self.get_model()


    def verify_params(self, params):
        self.logger.debug(f"Verify parameters for {self.model_name}.")

        # Default parameters.
        self.use_box_cox_transformer = False
        self.use_date_featurizer = False
        self.with_day_of_week = False
        self.with_day_of_month = False
        self.stepwise = True

        for param in params:
            if param == 'use_box_cox_endog_transformer':
                self.use_box_cox_transformer = bool(params[param])
            elif param == 'use_date_featurizer':
                self.use_date_featurizer = bool(params[param])
            elif param == 'date_featurizer_with_day_of_week':
                self.with_day_of_week = bool(params[param])
            elif param == 'date_featurizer_with_day_of_month':
                self.with_day_of_month = bool(params[param])
            elif param == 'stepwise':
                self.stepwise = bool(params[param])

        # Verify parameters
        if self.use_date_featurizer and self.modeling.dataset.discrete_interval != 'day':
            self.logger.error(f"Wrong parameters. DataFeaturizer cannot be used as discrete_interval not equal 'day'.")
            raise Exception

        if self.use_date_featurizer and self.modeling.dataset.interval_f is None:
            self.logger.error(f"Wrong parameters. DataFeaturizer cannot be used as interval_f is not specified.")
            raise Exception


    def get_model(self):

        self.logger.debug(f"Get model.")
        
        # Check parameters and create corresponding pipeline
        steps = list()

        if self.use_box_cox_transformer:
            bc_transformer = preprocessing.BoxCoxEndogTransformer(lmbda2=1e-6)
            steps.append(('step_bc', bc_transformer))
        
        if self.use_date_featurizer:
            date_featurizer = preprocessing.DateFeaturizer(
                column_name=self.modeling.dataset.interval_f,  # the name of the date feature in the X matrix
                with_day_of_week=self.with_day_of_week,
                with_day_of_month=self.with_day_of_month)
            steps.append(('step_df', date_featurizer))
        
        n_diffs = arima.ndiffs(self.modeling.y_train, max_d=5)

        steps.append(('step_arima', arima.AutoARIMA(d=n_diffs, trace=3,
            stepwise=self.stepwise,
            suppress_warnings=True,
            seasonal=True)))

        self.model = pipeline.Pipeline(steps)

    
    def fit(self):

        # Fit the pipeline in the appropriate way at the train part
        if self.use_date_featurizer:
            self.model.fit(self.modeling.y_train, self.modeling.X_train)
        
        else:
            self.model.fit(self.modeling.y_train)

        # Prediction for the test part.
        self.y_pred = self.model.predict(X=self.modeling.X_test, n_periods=self.modeling.n_intervals_estimation)

        # Update the pipeline with the test part 
        if self.use_date_featurizer:
            self.model.update(self.modeling.y_test, self.modeling.X_test)
        
        else:
            self.model.update(self.modeling.y_test)







class HoltWintersModel(BaseModel):

    def __init__(self):
        super().__init__()

class FbprophetsModel(BaseModel):

    def __init__(self):
        super().__init__()


# class ModelEstimator(object):

#     def __init__(self, models):
#         super().__init__()


#         self.models = models

#     def estimate(self):
#         pass

# class Estimator(object):

#     def __init__(self, n_intervals_estimation=28):
#         super().__init__()
#         pass

