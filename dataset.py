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
        target_f:
            Must be column name with target values.
        discrete_interval:
            A marker of what each line presents. Can be one of 'day', 'week, 'month', 'period'.
        interval_f:
            Must be None or column name with interval labels.
        dimension_f:
            Must be None or column name with dimension.
        target_f
    """

    def __init__(self, filename, target_f, discrete_interval, interval_f=None, dimension_f=None):

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

        if isinstance(dimension_f, list):
            self.logger.error(f"Parameter dimension_f must represent a single field! (Multi-dimension will be supported later)")
            raise Exception
        
        self.interval_f = interval_f
        self.dimension_f = dimension_f
        self.target_f = target_f

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

        if self.target_f not in self.data.columns:
            cols.append(self.target_f)

        if isinstance(self.dimension_f, str):
            if self.dimension_f not in self.data.columns:
                cols.append(self.dimension_f)

        if isinstance(self.dimension_f, list):
            for col in self.dimension_f:
                if col not in self.data.columns:
                    cols.append(col)
        
        if self.interval_f is not None and self.interval_f not in self.data.columns:
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

        if self.interval_f is None:
            # No column that presents interval name (date, week, month, period, etc.)
            self.logger.debug(f"A column with interval's labels is not specified.")
            self.logger.debug(f"It is assumed that the each dataset line represents a certain period of time")
            self.logger.debug(f"and all lines are listed in the correct order with no gaps.")
        
        else:
            
            if self.discrete_interval == 'day':
                # Each line in dataset should represens the date. A field with interval's labels is specified.
                self.logger.debug(f"Parameter discrete_interval is 'day' and a column with interval's labels is specified.")
                self.check_interval_is_date()
                self.expand_timeline()
            
            elif self.discrete_interval in ['week', 'month', 'period']:
                # Each line presents week or month or arbitrary period. Data gaps are not checked
                # If self.interval_f is specified, it can be used as labels 
                self.logger.debug(f"Parameter discrete_interval is '{self.discrete_interval}' and a column with interval's labels is specified.")
            
            else:
                self.logger.error(f"Unknown discrete interval {self.discrete_interval}!")
                raise Exception

        self.logger.debug(f"Intervals are verified.")


    def check_interval_is_date(self):
        """
        Runs in case when discrete_interval == 'day' and interval_f is specified.
        We have to be sure that the interval_f field consists of dates.

        !!! The function should use more durable way to check if interval_f consists of the dates.
        """
        self.data[self.interval_f] = pd.to_datetime(self.data[self.interval_f])
        pass

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

        min_date = min(self.data[self.interval_f])
        max_date = max(self.data[self.interval_f])
        # Arrange an array of datetime elements and convert to DataFrame
        tl = pd.DataFrame(np.arange(min_date, max_date, timedelta(days=1)).astype(datetime))
        tl.columns = [self.interval_f]
        tl.set_index(self.interval_f, inplace=True)
        
        exp_df = pd.DataFrame()

        if self.dimension_f is None:
            self.logger.debug(f"dimension_f is None.")
            ts = self.data.copy()
            ts.set_index(self.interval_f, inplace=True)
            ts = tl.merge(ts, how='left', left_index=True, right_index=True)
            ts[self.target_f].fillna(0, inplace=True)
            exp_df = pd.concat([exp_df, ts], axis=0)
            exp_df.reset_index(inplace=True)

        elif isinstance(self.dimension_f, str):
            self.logger.debug(f"dimension_f is '{self.dimension_f}'.")
            for dim in self.data[self.dimension_f].unique():
                ts = self.data[self.data[self.dimension_f]==dim].copy()
                ts.set_index(self.interval_f, inplace=True)
                ts = tl.merge(ts, how='left', left_index=True, right_index=True)
                ts[self.dimension_f].fillna(dim, inplace=True)
                ts[self.target_f].fillna(0, inplace=True)
                exp_df = pd.concat([exp_df, ts], axis=0)
            exp_df.reset_index(inplace=True)
            exp_df[self.dimension_f] = exp_df[self.dimension_f].astype(int)

        elif isinstance(self.dimension_f, list):
            self.logger.debug(f"dimension_f is '{', '.join(self.dimension_f)}'.")
            self.logger.error(f"Several column dimension is not supported!")
            raise Exception

        self.data = exp_df

        self.logger.debug(f"Timeline is expanded.")
