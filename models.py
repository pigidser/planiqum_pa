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

        self.model_selected = False
        self.model = None
        self.params_list = None
        model_selectors = {'arima': ArimaSelector, 'holtwinters': HoltWintersSelector, 'prophet': ProphetSelector}

        selector = model_selectors[model_type]
        self.model = selector(modeling, model_type, model_params, model_name)
            
        if self.model.get_best_model():
            self.model.fit()
            self.model_selected = True



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
        self.estimators = {'rmse': self.rmse_score, 'smape': self.smape_score}


    def dataset_adjustment(self):
        self.logger.debug(f"Dataset adjustment.")

        self.y_train = self.modeling.train[self.modeling.dataset.target_f].values
        self.y_test = self.modeling.test[self.modeling.dataset.target_f].values


    def get_model(self):
        pass


    def get_best_model(self):

        self.logger.debug(f"Seeking the best {self.model_type} model from {len(self.params_list)} variants.")
        
        found = False
        least_error = float(Inf)

        for params in self.params_list:
            
            self.logger.debug(f"Checking parameters: {params}")

            try:
                candidate = self.get_model(params, 'train')
                # Prediction for the test part.
                y_pred = self.predict(candidate, params)
                error = self.get_estimation(self.y_test, y_pred)

                if np.isnan(error):
                    self.logger.warning(f"Cannot estimate a model.")

                elif error < least_error:
                    least_error = error
                    best_params = params
                    best_y_pred = y_pred
                    best_model = candidate
                    found = True
                    self.logger.debug(f"Found a better model.")

            except Exception as e:
                self.logger.error(e)

        if found:
            self.best_y_pred = best_y_pred
            self.best_params = best_params
            self.best_model = best_model

            self.logger.debug(f"Found the best model with parameters: {self.best_params}.")

        else:
            self.logger.warning(f"Cannot create a model. Bad time series or try other parameters.")

        return found


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


    def get_estimation(self, actuals, forecast):

        estimations = []

        for metric in self.modeling.metric_list:

            f = self.estimators[metric]
            e = f(actuals, forecast)
            if not np.isnan(e):
                estimations.append(e)

        if len(estimations) == 0:
            return np.nan
        
        else:
            return np.sum(estimations) / len(estimations)


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
        self.dataset_adjustment()


    def verify_params(self, params):
        self.logger.debug(f"Verify parameters for {self.model_name}.")

        # Default parameters.
        use_boxcox = [True, False]
        use_date_featurizer = [True, False]
        with_day_of_week = [True, False]
        with_day_of_month = [True, False]
        stepwise = [True]
        m = [1]

        for param in params:
            
            if param == 'use_boxcox':
                use_boxcox = [bool(params[param])]
            
            elif param == 'use_date_featurizer':
                use_date_featurizer = [bool(params[param])]
            
            elif param == 'with_day_of_week':
                with_day_of_week = [bool(params[param])]
            
            elif param == 'with_day_of_month':
                with_day_of_month = [bool(params[param])]
            
            elif param == 'stepwise':
                stepwise = [bool(params[param])]

            elif param == 'm':
                m = [int(params[param])]

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

        # All combination of parameters for brute force
        for s in stepwise:
            for m1 in m:
                for b in use_boxcox:
                    for d in use_date_featurizer:
                        if not d:
                            # "with_day_" parameters make sense with Date Featurizer only.
                            d_w, d_m = False, False
                            self.params_list.append({
                                'use_boxcox': b, 'use_date_featurizer': d, 'with_day_of_week': d_w,
                                'with_day_of_month': d_m, 'stepwise': s, 'm': m1})
                        else:
                            for d_w in with_day_of_week:
                                for d_m in with_day_of_month:
                                    self.params_list.append({
                                        'use_boxcox': b, 'use_date_featurizer': d, 'with_day_of_week': d_w,
                                        'with_day_of_month': d_m, 'stepwise': s, 'm': m1})


    def dataset_adjustment(self):
        super().dataset_adjustment()
        
        # Arima requires X_train with timedate stamp for the Date Featurizer
        if self.modeling.dataset.interval_f is None:
            self.X_train, self.X_test = None, None
        else:
            # X_ part available only when column interval_f is specified 
            self.X_train = self.modeling.train[[self.modeling.dataset.interval_f]]
            self.X_test = self.modeling.test[[self.modeling.dataset.interval_f]]    


    def get_model(self, params, mode):

        use_boxcox = params['use_boxcox']
        use_date_featurizer = params['use_date_featurizer']
        with_day_of_week = params['with_day_of_week']
        with_day_of_month = params['with_day_of_month']
        stepwise = params['stepwise']
        m = params['m']

        if mode == 'full':
            y = np.concatenate([self.y_train, self.y_test])
            X = np.concatenate([self.X_train, self.X_test])
        elif mode == 'train':
            y = self.y_train
            X = self.X_train
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
            seasonal=True,
            m=m)))

        model = pipeline.Pipeline(steps)

        # Fit the pipeline in the appropriate way at the train part
        if use_date_featurizer:
            model.fit(y, X)
        
        else:
            model.fit(y)

        return model


    def predict(self, model, params):
        super().predict()
        return model.predict(X=self.X_test, n_periods=self.modeling.n_intervals_estimation)

    
    def fit(self):

        # Update the pipeline with the test part 
        if self.best_params['use_date_featurizer']:
            self.best_model.update(self.y_test, self.X_test)
        
        else:
            self.best_model.update(self.y_test)



class HoltWintersSelector(BaseSelector):

    def __init__(self, modeling, model_type, model_params, model_name):
        super().__init__()
        self.modeling = modeling
        self.model_type = model_type
        self.model_name = model_name
        self.verify_params(model_params)
        self.dataset_adjustment()


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
                seasonal_periods = [int(params[param])]
            
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


    def dataset_adjustment(self):
        super().dataset_adjustment()

        # if self.modeling.dataset.interval_f is None:
        #     self.X_train, self.X_test = None, None
        # else:
        #     # X_ part available only when column interval_f is specified 
        #     self.X_train = self.train[[self.modeling.dataset.interval_f]]
        #     self.X_test = self.test[[self.modeling.dataset.interval_f]]


    def get_model(self, params, mode):

        trend = params['trend']
        damped_trend = params['damped_trend']
        seasonal = params['seasonal']
        seasonal_periods = params['seasonal_periods']
        use_boxcox = params['use_boxcox']
        remove_bias = params['remove_bias']

        if mode == 'full':
            y = np.concatenate([self.y_train, self.y_test])
        elif mode == 'train':
            y = self.y_train
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


    def predict(self, model, params):
        super().predict()
        return model.forecast(steps=self.modeling.n_intervals_estimation)

    
    def fit(self):

        self.best_model = self.get_model(self.best_params, 'full')



class ProphetSelector(BaseSelector):

    def __init__(self, modeling, model_type, model_params, model_name):
        super().__init__()
        self.modeling = modeling
        self.model_type = model_type
        self.model_name = model_name
        self.verify_params(model_params)
        self.dataset_adjustment()


    def verify_params(self, params):
        self.logger.debug(f"Verify parameters for {self.model_name}.")

        # Default parameters.
        growth = ['linear']
        yearly_seasonality = [True, False]       # Fit yearly seasonality.
        weekly_seasonality = [True, False]       # Fit weekly seasonality.
        daily_seasonality = [True, False]        # Fit daily seasonality.
        seasonality_mode = ['additive', 'multiplicative']
        freq = None

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

            elif param == 'freq':
                freq = params[param]

            else:
                self.warning_param_unknown(param)

        if freq is None:
            if self.modeling.dataset.discrete_interval == 'day':
                freq = 'D'
            else:
                self.logger.error(f"The frequency parameter is mandatory if the discrete interval is not equal to 'day'.")
                self.logger.info(f"Example of freq: 'D' for day, 'MS' for month start, 'M' for month end.")
                self.logger.info(f"https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases")
                raise Exception

        # All models for brute force
        for g in growth:
            for y in yearly_seasonality:
                for w in weekly_seasonality:
                    for d in daily_seasonality:
                        for s in seasonality_mode:
                            self.params_list.append({
                                'growth': g, 'yearly_seasonality': y, 'weekly_seasonality': w,
                                'daily_seasonality': d, 'seasonality_mode': s, 'freq': freq})


    def dataset_adjustment(self):
        super().dataset_adjustment()
        
        if self.modeling.dataset.interval_f is None:
            self.logger.error(f"Prophet requires an interval field that can be cast to DateTime.")
            raise Exception
            
        # Prophet requires dataframe with ds and y columns
        self.ds_y_train = self.modeling.train[[self.modeling.dataset.interval_f, self.modeling.dataset.target_f]]
        self.ds_y_train.columns = ['ds', 'y']
        self.ds_y_test = self.modeling.test[[self.modeling.dataset.interval_f, self.modeling.dataset.target_f]]
        self.ds_y_test.columns = ['ds', 'y']


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
            df = pd.concat([self.ds_y_train, self.ds_y_test])
        elif mode == 'train':
            df = self.ds_y_train
        else:
            self.logger.error(f"The 'mode' parameter should be 'full' or 'train'!")
            raise Exception

        model.fit(df)

        return model


    def predict(self, model, params):
        super().predict()

        freq = params['freq']
        future = model.make_future_dataframe(periods=self.modeling.n_intervals_estimation, freq=freq)
        forecast = model.predict(future)
        
        return forecast['yhat'][-self.modeling.n_intervals_estimation:]

    
    def fit(self):

        self.best_model = self.get_model(self.best_params, 'full')


