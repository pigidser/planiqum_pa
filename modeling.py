from typing import Dict

from numpy.lib.npyio import save
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
    def __init__(self, dataset, folder, n_intervals_estimation, save_mode='all'):
        super().__init__()
        self.logger = logging.getLogger('planiqum_predictive_analitics.' + __name__)
        self.dataset = dataset
        self.define_dimension_values()
        self.load_model_base(folder)
        self.n_intervals_estimation = n_intervals_estimation
        if save_mode not in ['all', 'best']:
            save_mode = 'best'
        self.save_mode = save_mode
        self.selectors = list()
        self.results = dict()
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


    def add_selector(self, selector_type, selector_init_params, selector_name='Noname'):

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
            })

        self.logger.debug(f"Selector '{selector_type}' is added with params {selector_init_params} and name '{selector_name}'.")
        
        return True

    def run(self):

        if len(self.selectors) == 0:
            self.logger.warning(f"No one selector defined. Use add_selector method to choose selector.")
            return

        self.logger.debug(f"Modeling started.")

        if self.dimension_values is None:
            self.set_dimension(None)
            self.fit_selectors()
        else:
            for value in self.dimension_values:
                self.set_dimension(value)
                # Apply selectors to find models for current dimension value
                self.fit_selectors()

        self.logger.debug(f"Modeling completed.")


    def set_dimension(self, dimension_value=None):
        self.logger.debug(f"Get time series for {'entire dataset' if dimension_value is None else dimension_value}.")

        if dimension_value is None:
            # The dimension column is not specified. In that case, it is considered
            # that the dataset contains the only dimension value.
            self.dimension_value = None
            self.ts = self.dataset.data
        else:
            # Get a part of dataset for the asked dimension_value.
            self.dimension_value = dimension_value
            self.ts = self.dataset.data[self.dataset.data[self.dataset.dimension_col]==self.dimension_value]
        

    def fit_selectors(self):
        """
        Create models for self.dimension_value with all selectors.

        """
        # Splitting for particular ts
        self.train_test_split()

        result = dict()
        model_list = list()
        # Find models with each selector.
        for selector in self.selectors:
            self.set_selector(selector)
            model_id = self.get_model_id()
            model = self.fit_current_selector()
            # Model result.
            result[self.selector_type] = dict()
            result[self.selector_type]['model_id'] = model_id
            result[self.selector_type]['model_type'] = self.selector_type
            result[self.selector_type]['model_name'] = model_id

            if not model is None:
                result[self.selector_type]['params'] = model.best_params
                result[self.selector_type]['y_pred'] = model.best_y_pred
                result[self.selector_type]['metric_name'] = self.metric
                result[self.selector_type]['metric_value'] = model.metric_value
                result[self.selector_type]['result'] = 'ok'
                # Retain data to define the best model for the current demension value.
                model_list.append((self.selector_type, model_id, model, model.metric_value))

            else:
                # The selector is unable to find a model with selected parameters.
                result[self.selector_type]['result'] = 'fail'

        # Return the best model.
        if len(model_list) > 0:
            sel, id, mod, val = zip(*model_list)
            m = np.argmax(val)
            best_selector_type = sel[m]
            best_model = mod[m]
            best_model_id = id[m]
            result[best_selector_type]['best_metric'] = 1

            # Add result
            self.results[self.dimension_value] = result

            # Save all or the best model only
            if self.save_mode == 'best':
                self.save_model(best_model_id, best_model)
            else:
                for sel, id, mod, val in model_list:
                    self.save_model(id, mod)

            self.logger.debug(f"Found the best model for dimension value '{self.dimension_value}'.")

        else:
            self.logger.error(f"Failed to find any model for dimension value '{self.dimension_value}'.")
        

    def set_selector(self, selector):
        self.selector = selector['selector']
        self.selector_type = selector['selector_type']
        self.selector_init_params = selector['selector_init_params']
        self.selector_name = selector['selector_name']


    def train_test_split(self):
        
        self.train = self.ts.iloc[:len(self.ts) - self.n_intervals_estimation]
        self.test = self.ts.iloc[-self.n_intervals_estimation:]
        
        self.logger.debug(f"Train test split. Train part len={len(self.train)}, test part len={len(self.test)}.")


    def get_model_id(self):
        """Combine unique model id."""
        return f"{self.dimension_value}_{self.selector_type}_{self.selector_name}"


    def model_exists(self, model_id):
        return os.path.isfile(model_id + '.pkl')


    def fit_current_selector(self):
        """
        Create a model for current dimension value with current selector.

        """
        # An appropriate selector returns a model-wrapper
        self.logger.debug(f"Find the best model for dimension value '{self.dimension_value}' with selector {self.selector_type}")
        model = self.selector(self, self.selector_init_params)
        
        if model.get_best_model():
            model.fit()
            return model

        else:
            self.logger.warning(f"Cannot select a model with parameters above.")
            return None


    def save_model(self, model_id, model):

        self.logger.debug(f"Save model {model_id}.")

        model_file_name = os.path.join(self.model_base_folder, f"{model_id}.pkl")
        plot_file_name = os.path.join(self.model_base_folder, f"{model_id}.png")

        self.to_png(model.y_train, model.y_test, model.best_y_pred, plot_file_name)
        self.to_pkl(model.best_model, model_file_name)


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
        