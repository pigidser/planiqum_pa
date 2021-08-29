# from typing import Dict

# from numpy.lib.function_base import select
import logging
import traceback
import os, sys
from time import time
import numpy as np
from numpy.core.numeric import Inf
import pandas as pd

# arima predictor
from pmdarima import arima
from pmdarima import pipeline
from pmdarima import preprocessing

# holt-winter's predictor
from statsmodels.tsa.api import ExponentialSmoothing

# fbprophet predictor
from fbprophet import Prophet

from utilities import *

# from modeling import Modeling
# from dataset import Dataset


class ModelSelector(object):

    def __init__(self, modeling, model_type, model_params, model_name):
        super().__init__()
        self.logger = logging.getLogger('planiqum_predictive_analitics.' + __name__)


        self.model = None
        self.params_list = None

        if model_type == 'arima':
            self.model = ArimaSelector(modeling, model_type, model_params, model_name)

        elif model_type == 'holtwinters':
            self.model = HoltWintersSelector(modeling, model_type, model_params, model_name)

        elif model_type == 'fbprophet':
            self.model = FbprophetSelector(modeling, model_type, model_params, model_name)

        else:
            self.logger.error(f"Unknown model type!")
            raise Exception
            
        self.model.get_best_model()
        self.model.fit()



class BaseSelector(object):

    def __init__(self):
        self.logger = logging.getLogger('planiqum_predictive_analitics.' + __name__)
        self.params_list = list()
        self.best_params = None
        self.best_model = None
        self.best_y_pred = None
        self.model_name = None
        self.model_type = None
        self.modeling = None


    def get_model(self):
        pass


    def get_best_model(self):

        self.logger.debug(f"Seeking the best {self.model_type} model from {len(self.params_list)} variants.")
        
        least_error = float(Inf)

        for params in self.params_list:
            
            self.logger.debug(f"Checking parameters: {params}")

            try:
                candidate = self.get_model(params, 'train')
                # Prediction for the test part.
                y_pred = self.predict(candidate)
                error = self.average_error(self.modeling.y_test, y_pred)

                if error < least_error:
                    least_error = error
                    best_params = params
                    best_y_pred = y_pred
                    best_model = candidate

            except Exception as e:
                self.logger.error(e)

        self.best_y_pred = best_y_pred
        self.best_params = best_params
        self.best_model = best_model

        self.logger.debug(f"Best parameters: {self.best_params}.")


    def fit(self):
        if self.best_model is None:
            self.logger.error(f"Method 'fit' run after 'get_best_model'!")
            raise Exception


    def predict(self):
        pass


    def smape_score(self, actuals, forecast):
        return (100/len(actuals) * np.sum(np.abs((forecast - actuals)) / ((np.abs(actuals) + np.abs(forecast)) / 2)))


    def rmse_score(self, actuals, forecast):
        return (np.sum(np.power((forecast - actuals), 2)) / len(actuals))**(1/2)


    def average_error(self, actuals, forecast):
        return 1/2 * (self.rmse_score(actuals, forecast) + self.smape_score(actuals, forecast))


    def warning_param_unknown(self, param_name):
        self.logger.warning(f"Parameter {param_name} is unknown.")


    def warning_param_bad_value(self, param_name, value):
        self.logger.warning(f"Parameter {param_name} value is incorrect.")



class ArimaSelector(BaseSelector):

    def __init__(self, modeling, model_type, model_params, model_name):
        super().__init__()
        self.modeling = modeling
        self.model_type = model_type
        self.model_name = model_name
        self.verify_params(model_params)


    def verify_params(self, params):
        self.logger.debug(f"Verify parameters for {self.model_name}.")

        # Default parameters.
        use_boxcox = [True, False]
        use_date_featurizer = [True, False]
        with_day_of_week = [True, False]
        with_day_of_month = [True, False]
        stepwise = [True, False]

        for param in params:
            
            if param == 'use_box_cox_endog_transformer':
                use_boxcox = [bool(params[param])]
            
            elif param == 'use_date_featurizer':
                use_date_featurizer = [bool(params[param])]
            
            elif param == 'date_featurizer_with_day_of_week':
                with_day_of_week = [bool(params[param])]
            
            elif param == 'date_featurizer_with_day_of_month':
                with_day_of_month = [bool(params[param])]
            
            elif param == 'stepwise':
                stepwise = [bool(params[param])]

            else:
                self.warning_param_unknown(param)

        # Verify parameters
        if self.modeling.dataset.discrete_interval != 'day':
            if True in use_date_featurizer:
                self.logger.warning(f"Wrong parameters. DataFeaturizer cannot be used if discrete_interval is not equal 'day'.")
                use_date_featurizer = [False]

        if self.modeling.dataset.interval_f is None:
            if True in use_date_featurizer:
                self.logger.warning(f"Wrong parameters. DataFeaturizer cannot be used as interval_f is not specified.")
                use_date_featurizer = [False]

        # All models for brute force
        for b in use_boxcox:
            for d in use_date_featurizer:
                for w in with_day_of_week:
                    for m in with_day_of_month:
                        for s in stepwise:
                            self.params_list.append({
                                'use_boxcox': b, 'use_date_featurizer': d, 'with_day_of_week': w,
                                'with_day_of_month': m, 'stepwise': s})


    def get_model(self, params, mode):

        use_boxcox = params['use_boxcox']
        use_date_featurizer = params['use_date_featurizer']
        with_day_of_week = params['with_day_of_week']
        with_day_of_month = params['with_day_of_month']
        stepwise = params['stepwise']

        if mode == 'full':
            y = np.concatenate([self.modeling.y_train, self.modeling.y_test])
            X = np.concatenate([self.modeling.X_train, self.modeling.X_test])
        elif mode == 'train':
            y = self.modeling.y_train
            X = self.modeling.X_train
        else:
            self.logger.error(f"The 'mode' parameter should be 'full' or 'train'!")
            raise Exception

        # Check parameters and create corresponding pipeline
        steps = list()

        if use_boxcox:
            bc_transformer = preprocessing.BoxCoxEndogTransformer(lmbda2=1e-6)
            steps.append(('step_bc', bc_transformer))
        
        if use_date_featurizer:
            date_featurizer = preprocessing.DateFeaturizer(
                column_name=self.modeling.dataset.interval_f,  # the name of the date feature in the X matrix
                with_day_of_week=with_day_of_week,
                with_day_of_month=with_day_of_month)
            steps.append(('step_df', date_featurizer))
        
        n_diffs = arima.ndiffs(y, max_d=5)

        steps.append(('step_arima', arima.AutoARIMA(d=n_diffs, trace=3,
            stepwise=stepwise,
            suppress_warnings=True,
            seasonal=True)))

        model = pipeline.Pipeline(steps)

        # Fit the pipeline in the appropriate way at the train part
        if use_date_featurizer:
            model.fit(y, X)
        
        else:
            model.fit(y)

        return model


    def predict(self, model):
        super().predict()
        return model.predict(X=self.modeling.X_test, n_periods=self.modeling.n_intervals_estimation)

    
    def fit(self):

        # Update the pipeline with the test part 
        if self.best_params['use_date_featurizer']:
            self.best_model.update(self.modeling.y_test, self.modeling.X_test)
        
        else:
            self.best_model.update(self.modeling.y_test)



class HoltWintersSelector(BaseSelector):

    def __init__(self, modeling, model_type, model_params, model_name):
        super().__init__()
        self.modeling = modeling
        self.model_type = model_type
        self.model_name = model_name
        self.verify_params(model_params)


    def verify_params(self, params):
        self.logger.debug(f"Verify parameters for {self.model_name}.")

        # Default parameters.
        trend = ['add', 'mul', None]       # Type of trend component.
        damped_trend = [True, False]       # Should the trend component be damped.
        seasonal = ['add', 'mul', None]    # Type of seasonal component.
        seasonal_periods = None            # The number of periods in a complete seasonal cycle
        use_boxcox = [True, False]         # Should the Box-Cox transform be applied to the data first?
        remove_bias = [True, False]        # Remove bias from forecast values

        for param in params:
            
            if param == 'trend':
                if params[param] in ['add', 'mul', 'additive', 'multiplicative', None]:
                    trend = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])
            
            elif param == 'damped_trend':
                damped_trend = [bool(params[param])]
            
            elif param == 'seasonal':
                if params[param] in ['add', 'mul', 'additive', 'multiplicative', None]:
                    seasonal = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])
            
            elif param == 'seasonal_periods':
                seasonal_periods = int(params[param])
            
            elif param == 'use_boxcox':
                use_boxcox = [bool(params[param])]

            elif param == 'remove_bias':
                remove_bias = [bool(params[param])]

            else:
                self.warning_param_unknown(param)

        # Verify parameters
        if seasonal_periods is None:
            if self.modeling.dataset.discrete_interval == 'day':
                seasonal_periods = [7]

            elif self.modeling.dataset.discrete_interval == 'month':
                seasonal_periods = [12]

            else:
                seasonal_periods = [7, 12, 4, 13]

        # All models for brute force
        for t in trend:
            for d in damped_trend:
                for s in seasonal:
                    for p in seasonal_periods:
                        for b in use_boxcox:
                            for r in remove_bias:
                                self.params_list.append(
                                    {'trend': t, 'damped_trend': d, 'seasonal': s, 'seasonal_periods': p,
                                     'use_boxcox': b, 'remove_bias': r})


    def get_model(self, params, mode):

        trend = params['trend']
        damped_trend = params['damped_trend']
        seasonal = params['seasonal']
        seasonal_periods = params['seasonal_periods']
        use_boxcox = params['use_boxcox']
        remove_bias = params['remove_bias']

        if mode == 'full':
            y = np.concatenate([self.modeling.y_train, self.modeling.y_test])
        elif mode == 'train':
            y = self.modeling.y_train
        else:
            self.logger.error(f"The 'mode' parameter should be 'full' or 'train'!")
            raise Exception

        if trend is None:
            model = ExponentialSmoothing(y, trend=trend,
                seasonal=seasonal, seasonal_periods=seasonal_periods, use_boxcox=use_boxcox,
                initialization_method='estimated')

        else:
            model = ExponentialSmoothing(y, trend=trend, damped_trend=damped_trend,
                seasonal=seasonal, seasonal_periods=seasonal_periods, use_boxcox=use_boxcox,
                initialization_method='estimated')

        fitted = model.fit(optimized=True, remove_bias=remove_bias)
        
        return fitted


    def predict(self, model):
        super().predict()
        return model.forecast(steps=self.modeling.n_intervals_estimation)

    
    def fit(self):

        # self.best_model = self.get_model(self.best_params, np.concatenate([self.modeling.y_train, self.modeling.y_test]))
        self.best_model = self.get_model(self.best_params, 'full')



class FbprophetSelector(BaseSelector):

    def __init__(self, modeling, model_type, model_params, model_name):
        super().__init__()
        self.modeling = modeling
        self.model_type = model_type
        self.model_name = model_name
        self.verify_params(model_params)


    def verify_params(self, params):
        self.logger.debug(f"Verify parameters for {self.model_name}.")

        # Default parameters.
        growth = ['linear']
        yearly_seasonality = [True, False]       # Fit yearly seasonality.
        weekly_seasonality = [True, False]       # Fit weekly seasonality.
        daily_seasonality = [True, False]        # Fit daily seasonality.
        seasonality_mode = ['additive', 'multiplicative']

        for param in params:
            
            if param == 'growth':
                if params[param] in ['linear', 'logistic']:
                    growth = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])
            
            elif param == 'yearly_seasonality':
                if params[param] in ['auto', True, False]:
                    yearly_seasonality = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])

            elif param == 'weekly_seasonality':
                if params[param] in ['auto', True, False]:
                    weekly_seasonality = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])
            
            elif param == 'daily_seasonality':
                if params[param] in ['auto', True, False]:
                    daily_seasonality = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])

            elif param == 'seasonality_mode':
                if params[param] in ['additive', 'multiplicative']:
                    seasonality_mode = [params[param]]
                else:
                    self.warning_param_bad_value(param, params[param])

            else:
                self.warning_param_unknown(param)

        # All models for brute force
        for g in growth:
            for y in yearly_seasonality:
                for w in weekly_seasonality:
                    for d in daily_seasonality:
                        for s in seasonality_mode:
                            self.params_list.append({
                                'growth': g, 'yearly_seasonality': y, 'weekly_seasonality': w,
                                'daily_seasonality': d, 'seasonality_mode': s})


    def get_model(self, params, mode):

        growth = params['growth']
        yearly_seasonality = params['yearly_seasonality']
        weekly_seasonality = params['weekly_seasonality']
        daily_seasonality = params['daily_seasonality']
        seasonality_mode = params['seasonality_mode']

        model = Prophet(growth=growth,
            changepoints=None, 
            n_changepoints=25,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality, 
            weekly_seasonality=weekly_seasonality, 
            daily_seasonality=daily_seasonality,
            holidays=None)

        if mode == 'full':
            df = pd.concat([self.modeling.ds_y_train, self.modeling.ds_y_test])
        elif mode == 'train':
            df = self.modeling.ds_y_train
        else:
            self.logger.error(f"The 'mode' parameter should be 'full' or 'train'!")
            raise Exception

        model.fit(df)

        return model


    def predict(self, model):
        super().predict()
        future = model.make_future_dataframe(periods=self.modeling.n_intervals_estimation)
        forecast = model.predict(future)
        return forecast['yhat'][-self.modeling.n_intervals_estimation:]

    
    def fit(self):

        # self.best_model = self.get_model(self.best_params, np.concatenate([self.modeling.ds_y_train, self.modeling.ds_y_test]))
        self.best_model = self.get_model(self.best_params, 'full')


