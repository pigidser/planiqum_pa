from typing import List
from utilities import *
import logging
import traceback
import os, sys

from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd


class Dataset(object):
    """
    Represent dataset and provide initial checks and transformations.
    Parameters
    ----------
        filename : string
            File name without path.
        target_col:
            Must be column name with target values.
        discrete_interval:
            A marker of what each line presents. Can be one of 'day', 'week, 'month', 'period'.
        date_col:
            Must be None or column name with interval labels.
        dimension_col:
            Must be None or column name with dimension.
        target_col
    """

    def __init__(self, filename, target_col, discrete_interval, date_col=None, dimension_col=None):

        super().__init__()
        self.logger = logging.getLogger('planiqum_predictive_analitics.' + __name__)
        if not os.path.isfile(filename):
            self.logger.error(f"Data file '{filename}' not found!")
            raise Exception

        self.filename = filename

        if discrete_interval not in ['day', 'week', 'month', 'period']:
            self.logger.error(f"Discrete interval must be one of 'day', 'week', 'month', 'period'!")
            raise Exception
        self.discrete_interval = discrete_interval

        if isinstance(dimension_col, list):
            self.logger.error(f"Parameter dimension_col must represent a single field! (Multi-dimension will be supported later)")
            raise Exception
        
        self.date_col = date_col
        self.dimension_col = dimension_col
        self.target_col = target_col

        self.load()
        self.check_columns_exist()
        self.verify_intervals()


    def load(self):
        self.data = pd.read_csv(self.filename)
        self.logger.debug(f"Data is loaded with length: {self.data.shape[0]}")


    def check_columns_exist(self):
        """
        Check if all columns exist in dataset.
        """
        cols = []

        if self.target_col not in self.data.columns:
            cols.append(self.target_col)

        if isinstance(self.dimension_col, str):
            if self.dimension_col not in self.data.columns:
                cols.append(self.dimension_col)

        if isinstance(self.dimension_col, list):
            for col in self.dimension_col:
                if col not in self.data.columns:
                    cols.append(col)
        
        if self.date_col is not None and self.date_col not in self.data.columns:
            cols.append(col)

        if len(cols) > 0:
            self.logger.error(f"Unknown columns: {', '.join(cols)}!")
            raise Exception

        self.logger.debug(f"All columns exist in dataset")


    def verify_intervals(self):
        """
        Check if interval
        """
        self.logger.debug(f"Verify intervals...")

        if self.date_col is None:
            # No column that presents date colunn
            self.logger.debug(f"A date column is not specified.")
            self.logger.debug(f"It is assumed that the each dataset line represents a certain period of time")
            self.logger.debug(f"and all lines are listed in the correct order with no gaps.")
        
        else:

            self.logger.debug(f"Date column '{self.date_col}' is specified.")            
            self.check_interval_is_date()

            self.logger.debug(f"Parameter discrete_interval is '{self.discrete_interval}'.")

            if self.discrete_interval == 'day':
                # Each line in dataset should represens the date. A field with interval's labels is specified.
                self.expand_timeline()
            
            elif self.discrete_interval in ['week', 'month', 'period']:
                # Each line presents week or month or arbitrary period. Data gaps are not checked
                pass
            
            else:
                self.logger.warning(f"Unknown discrete interval {self.discrete_interval}!")

        self.logger.debug(f"The column with interval labels is verified.")


    def check_interval_is_date(self):
        """
        Runs when date_col is specified to check if it consists of dates.

        !!! The function should use more durable way to check if date_col consists of the dates.
        """
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        

    def expand_timeline(self):
        """
        Define the minimal and maximal dates in the input time series and fill skipped dates for each product.
        
        Parameters
        ----------
        ts : DataFrame
            Time series data of which to expand.
            
        Returns
        -------
        DataFrame
            Expanded time series.
        
        """
        if self.data.empty:
            self.logger.error(f"Envoke the load method first!")
            raise Exception

        if not self.discrete_interval == 'day':
            return

        min_date = min(self.data[self.date_col])
        max_date = max(self.data[self.date_col])

        interval_td = timedelta(days=1)

        # Timeline template
        tl = pd.DataFrame(np.arange(min_date, max_date + interval_td, interval_td).astype(datetime), columns=[self.date_col])

        if self.dimension_col is None:
            self.logger.debug(f"Dimension field is not specified.")
            exp_ts = tl.merge(self.data, on=[self.date_col], how='left').fillna(0)

        elif isinstance(self.dimension_col, str):
            self.logger.debug(f"dimension_col is '{self.dimension_col}'.")

            # Product template
            dimensions = pd.DataFrame(self.data[self.dimension_col].drop_duplicates(), columns=[self.dimension_col])
            dimensions['dummy'] = np.NaN

            tl['dummy'] = np.NaN

            # Interval/dimension template
            tmpl = tl.merge(dimensions, on='dummy', how='outer').drop('dummy', axis='columns')
            exp_ts = tmpl.merge(self.data, on=[self.dimension_col, self.date_col], how='left').fillna(0)

        elif isinstance(self.dimension_col, list):
            self.logger.debug(f"dimension_col is '{', '.join(self.dimension_col)}'.")
            self.logger.error(f"Several column dimension is not supported!")
            raise Exception

        self.data = exp_ts

        self.logger.debug(f"Timeline is expanded.")
