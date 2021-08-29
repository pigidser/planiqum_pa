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
from models import ModelSelector


class Modeling(object):
    """
    Build and estimate models.

    """
    def __init__(self, dataset, folder, n_intervals_estimation, dimension_values=None):
        super().__init__()
        self.logger = logging.getLogger('planiqum_predictive_analitics.' + __name__)
        self.dataset = dataset
        self.load_model_base(folder)
        self.n_intervals_estimation = n_intervals_estimation
        self.define_dimension_values(dimension_values)
        self.model_list = []
        self.metric_list = []
        

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


    def define_dimension_values(self, dimension_values):
        self.logger.debug(f"Define dimension values.")

        if self.dataset.dimension_f is None:
            self.logger.debug(f"Column dimension_f is not specified, so dataset consists of the one dimension value.")
            self.dimension_values = None

        else:
            self.logger.debug(f"Column dimension_f is specified. Create a list of dimension values for modeling.")
            all_dimension_values = self.dataset.data[self.dataset.dimension_f].unique()
            
            if dimension_values is None:
                self.dimension_values = all_dimension_values
                self.logger.debug(f"Parameter dimension_values is None, so take all {len(self.dimension_values)} dimensions for modeling.")

            else:
                self.logger.debug(f"Parameter dimension_values is specified with length {len(dimension_values)}.")
                self.dimension_values = [x for x in dimension_values if x in all_dimension_values]

                if len(self.dimension_values) == 0:
                    self.dimension_values = all_dimension_values
                    self.logger.debug(f"Parameter dimension_values consists of unknown values. All {len(self.dimension_values)} existing values will be taken for modeling.")


    def define_metrics(self, metrics):
        """
        Define a list of metrics that will be used for estimation.
        Parameters
        ----------
            metrics : string or list of strings
                Metrics.
        """

        if isinstance(metrics, str):
            metrics = [metrics]

        if isinstance(metrics, list):
            for metric in metrics:
                if metric in ['rmse', 'smape']:
                    self.metric_list.append(metric)
                else:
                    self.logger.warning(f"Unknown or unsupported metric '{metric}'")


    def add_model(self, model_type, model_params, model_name='best_model'):

        if model_type not in ['arima', 'holtwinters', 'prophet']:
            self.logger.warning("Model type {model_type} is not supported.")
            return False

        if not isinstance(model_params, dict):
            self.logger.warning("Model params must be dictionary.")
            return False

        self.model_list.append((model_type, model_params, model_name))
        self.logger.debug(f"The model '{model_name} is added with params {model_params}.")
        
        return True

    def run_modeling(self, recalculate=True):

        if len(self.model_list) == 0:
            self.logger.warning(f"Model list is empty. Use add_model method to create a list of models.")
            return

        if len(self.metric_list) == 0:
            self.logger.warning(f"Metric list is empty. Use define_metric method to add one or several metric.")
            return

        self.logger.debug(f"Run modeling.")

        self.recalculate = recalculate

        if self.dimension_values is None:
            self.get_time_series()
            self.create_models()
        else:
            for value in self.dimension_values:
                self.get_time_series(value)
                self.create_models()


    def get_time_series(self, dimension_value=None):
        self.logger.debug(f"Get time series for {'entire dataset' if dimension_value is None else dimension_value}.")

        if dimension_value is None:
            # Get all dataset as consisting the only dimension value.
            self.cur_dim_val = None
            self.cur_ts = self.dataset.data
        else:
            # Get a part of dataset for the asked dimension_value.
            self.cur_dim_val = dimension_value
            self.cur_ts =  self.dataset.data[self.dataset.data[self.dataset.dimension_f]==self.cur_dim_val]
        

    def create_models(self):
        self.train_test_split()
        for model in self.model_list:
            self.select_model(model)


    def train_test_split(self):
        
        self.train = self.cur_ts.iloc[:len(self.cur_ts) - self.n_intervals_estimation]
        self.test = self.cur_ts.iloc[-self.n_intervals_estimation:]
        
        self.logger.debug(f"Train test split. Train part len={len(self.train)}, test part len={len(self.test)}.")


    def select_model(self, model):

        model_type = model[0]
        model_parameters = model[1]
        model_name = model[2]
        model_file_name = os.path.join(self.model_base_folder, f"{self.cur_dim_val}_{model_type}_{model_name}.pkl")
        plot_file_name = os.path.join(self.model_base_folder, f"{self.cur_dim_val}_{model_type}_{model_name}.png")

        # if self.select_model(self.cur_dim_val, model_name) and not self.recalculate:
        if os.path.isfile(model_file_name) and not self.recalculate:
            self.logger.debug(f"The model {model_name} exists for dimension value {self.cur_dim_val} and recalculation is not requested.")
            return

        # Find the best model and fit
        selector = ModelSelector(self, model_type, model_parameters, model_name)

        if selector.model_selected:
            self.logger.debug(f"SMAPE = {selector.model.smape_score(selector.model.y_test, selector.model.best_y_pred)}")
            self.logger.debug(f"RMSE = {selector.model.rmse_score(selector.model.y_test, selector.model.best_y_pred)}")

            self.save_plot(selector.model.y_train, selector.model.y_test, selector.model.best_y_pred, plot_file_name)
            self.save_model(selector.model.best_model, model_file_name)

        else:
            self.logger.warning(f"Cannot select a model with parameters above.")


    def save_model(self, model, model_file_name):
        # Save the model
        joblib.dump(model, model_file_name, compress=3)
    

    def save_plot(self, y_train, actuals, forecast, plot_file_name):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)

        n_train = y_train.shape[0]
        x = np.arange(n_train + forecast.shape[0])

        ax.plot(x[:n_train], y_train, color='blue', label='Training Data')
        ax.plot(x[n_train:], forecast, color='green', marker='o',
                label='Predicted')
        ax.plot(x[n_train:], actuals, color='red', label='Actual')
        ax.legend(loc='lower left', borderaxespad=0.5)
        ax.set_title(f"Actuals vs Forecast for dimension value {self.cur_dim_val}")
        ax.set_ylabel('Sold units')
        plt.savefig(plot_file_name)





