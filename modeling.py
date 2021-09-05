from typing import Dict
from utilities import *
import logging
import traceback
import os, sys
import joblib
from time import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from dataset import Dataset
from models import ArimaSelector, HoltWintersSelector, ProphetSelector

selector_list = {'arima': ArimaSelector, 'holtwinters': HoltWintersSelector, 'prophet': ProphetSelector}


class Modeling(object):
    """
    Build and estimate models.

    """
    def __init__(self, dataset, folder, n_intervals_estimation):
        super().__init__()
        self.logger = logging.getLogger('planiqum_predictive_analitics.' + __name__)
        self.dataset = dataset
        self.define_dimension_values()
        self.load_model_base(folder)
        self.n_intervals_estimation = n_intervals_estimation
        self.selectors = list()
        self.models = list()
        self.metric = 'rmse'
        

    def load_model_base(self, folder):

        self.model_base_folder = os.path.join(os.getcwd(), folder)
        self.model_base_file = os.path.join(self.model_base_folder, 'database.csv')
        
        if not os.path.isdir(self.model_base_folder):
            os.makedirs(self.model_base_folder)
            self.model_base = pd.DataFrame()
        
        elif os.path.isfile(self.model_base_file):
            self.model_base = pd.read_csv()


    def select_model(self, dimension_value, model_name):

        return len(self.model_base[(self.model_base['dim_value']==dimension_value) & (self.model_base['model_name']==model_name)]) == 1
        

    def insert_model(self, dimension_value, model_name, model_parameters, estimator):

        pass


    def define_dimension_values(self, dimension_values=None):
        self.logger.debug(f"Define dimension values.")

        if self.dataset.dimension_col is None:
            self.logger.debug(f"Column dimension_col is not specified, so dataset consists of the one dimension value.")
            self.dimension_values = None

        else:
            self.logger.debug(f"Column dimension_col is specified. Create a list of dimension values for modeling.")
            all_dimension_values = self.dataset.data[self.dataset.dimension_col].unique()
            
            if dimension_values is None:
                self.dimension_values = all_dimension_values
                self.logger.debug(f"Parameter dimension_values is None, so take all {len(self.dimension_values)} dimensions for modeling.")

            else:
                self.logger.debug(f"Parameter dimension_values is specified with length {len(dimension_values)}.")
                self.dimension_values = [x for x in dimension_values if x in all_dimension_values]

                if len(self.dimension_values) == 0:
                    self.dimension_values = all_dimension_values
                    self.logger.debug(f"Parameter dimension_values consists of unknown values. All {len(self.dimension_values)} existing values will be taken for modeling.")


    def set_metric(self, metric):
        """
        Define a metric for estimation of the model.
        Parameters
        ----------
            metric : string
        """

        if isinstance(metric, str):
            if metric in ['rmse', 'smape']:
                self.metric = metric
            else:
                self.logger.warning(f"Unknown or unsupported metric '{metric}'")
        else:
            self.logger.warning(f"Define a metric as string name")


    def add_selector(self, selector_type, selector_init_params, selector_name='Noname', recalculate=True):

        if selector_type not in selector_list.keys():
            self.logger.warning("Selector type {model_type} is not supported.")
            return False
        else:
            selector = selector_list[selector_type]

        if not isinstance(selector_init_params, dict):
            self.logger.warning("Selector initial params must be dictionary.")
            return False

        self.selectors.append({
            'selector': selector,
            'selector_type': selector_type,
            'selector_init_params': selector_init_params,
            'selector_name': selector_name,
            'recalculate': recalculate,
            })

        self.logger.debug(f"Selector '{selector_type}' is added with params {selector_init_params} and name '{selector_name}'.")
        
        return True

    def start_modeling(self):

        if len(self.selectors) == 0:
            self.logger.warning(f"No one selector defined. Use add_selector method to choose one or several selectors.")
            return

        self.logger.debug(f"Start modeling.")

        if self.dimension_values is None:
            self.get_time_series()
            self.create_all_models()
        else:
            for value in self.dimension_values:
                self.get_time_series(value)
                # Create models for defined dimension values
                self.create_all_models()


    def get_time_series(self, dimension_value=None):
        self.logger.debug(f"Get time series for {'entire dataset' if dimension_value is None else dimension_value}.")

        if dimension_value is None:
            # Get all dataset as consisting the only dimension value.
            self.dimension_value = None
            self.ts = self.dataset.data
        else:
            # Get a part of dataset for the asked dimension_value.
            self.dimension_value = dimension_value
            self.ts = self.dataset.data[self.dataset.data[self.dataset.dimension_col]==self.dimension_value]
        

    def create_all_models(self):
        """
        Create models for self.dimension_value with all selectors.

        """
        # Splitting for particular ts
        self.train_test_split()
        # With each selector create models  
        for selector in self.selectors:
            self.selector = selector['selector']
            self.selector_type = selector['selector_type']
            self.selector_init_params = selector['selector_init_params']
            self.selector_name = selector['selector_name']
            self.selector_recalculate = selector['recalculate']
            self.create_model()


    def train_test_split(self):
        
        self.train = self.ts.iloc[:len(self.ts) - self.n_intervals_estimation]
        self.test = self.ts.iloc[-self.n_intervals_estimation:]
        
        self.logger.debug(f"Train test split. Train part len={len(self.train)}, test part len={len(self.test)}.")


    def get_model_id(self):
        return f"{self.dimension_value}_{self.selector_type}_{self.selector_name}"


    def model_exists(self):
        return os.path.isfile(self.get_model_id() + '.pkl')


    def create_model(self):
        """
        Create a model for current dimension value with current selector.

        """
        if self.model_exists and not self.selector_recalculate:
            self.logger.debug(f"The model {self.get_model_id()} exists for dimension value {self.dimension_value} and recalculation is not requested.")
            return

        # An appropriate selector returns a model-wrapper
        self.logger.debug(f"Find the best model for dimension value '{self.dimension_value}' with selector {self.selector_type}")
        model = self.selector(self, self.selector_init_params)
        
        if model.get_best_model():
            model.fit()
            self.save_model(model)

        else:
            self.logger.warning(f"Cannot select a model with parameters above.")


    def save_model(self, model):

        model_id = self.get_model_id()
        model_file_name = os.path.join(self.model_base_folder, f"{model_id}.pkl")
        plot_file_name = os.path.join(self.model_base_folder, f"{model_id}.png")

        self.logger.debug(f"Saving result for {model_id} ...")

        smape = model.smape_score(model.y_test, model.best_y_pred)
        rmse = model.rmse_score(model.y_test, model.best_y_pred)

        self.logger.debug(f"SMAPE = {smape}, RMSE = {rmse}")

        self.to_png(model.y_train, model.y_test, model.best_y_pred, plot_file_name)
        self.to_pkl(model.best_model, model_file_name)

        result = {
            'id': model_id,
            'dimension_value': self.dimension_value,
            'model_type': self.selector_type,
            'model_name': model_id,
            'params': model.best_params,
            'y_pred': model.best_y_pred,
            'rmse': rmse,
            'smape': smape,
            }
        self.models.append(result)


    def to_pkl(self, model, model_file_name):
        # Save the model
        joblib.dump(model, model_file_name, compress=3)
    

    def to_png(self, y_train, actuals, forecast, plot_file_name):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)

        n_train = y_train.shape[0]
        x = np.arange(n_train + forecast.shape[0])

        ax.plot(x[:n_train], y_train, color='blue', label='Training Data')
        ax.plot(x[n_train:], forecast, color='green', marker='o',
                label='Predicted')
        ax.plot(x[n_train:], actuals, color='red', label='Actual')
        ax.legend(loc='lower left', borderaxespad=0.5)
        ax.set_title(f"Actuals vs Forecast for dimension value {self.dimension_value}")
        ax.set_ylabel('Sold units')
        plt.savefig(plot_file_name)
        