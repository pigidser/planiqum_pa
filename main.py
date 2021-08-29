from utilities import *
import logging
import os, sys
from time import time
import argparse

from dataset import Dataset
from modeling import Modeling

def main():

    try:
        t0 = time()
        # Set up logging
        logger = logging.getLogger('planiqum_predictive_analitics')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(u"./logs/planiqum_pa.log", "w")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.debug("Parsing command line")

        # parser = argparse.ArgumentParser(description="Planiqum forecast module")

        # # Parse arguments
        # parser.add_argument("-p", "--operation", type=str, action='store',
        #     help="Please specify the operation to execute", required=False, default='None')
        # parser.add_argument("-c", "--data", type=str, action='store',
        #     help="Please specify a data file with time series", required=False, default='orders.csv')
        # parser.add_argument("-r", "--output", type=str, action='store',
        #     help="Please specify a folder to store models", required=False, default='pre-calc-models')

        # args = parser.parse_args()

        # if args.operation != 'None':
        #     operation = args.operation
        #     data_file = args.data
        #     output_folder = args.output

        operation = 'estimation'
        data_file = 'orders.csv'

        # Initialize
        if operation.lower() == 'estimation':
            logger.info(f"Initialization of time series data from {data_file}")
                
            data = Dataset(
                filename='./data/orders - product 2354.csv',
                target_f='Quantity',
                discrete_interval='day',
                interval_f='Date',
                dimension_f='ProductId',
                )

            modeling = Modeling(data, "model_base", n_intervals_estimation=28)

            modeling.add_model('arima', {
                'use_box_cox_endog_transformer' : True,
                'use_date_featurizer': True,
                'date_featurizer_with_day_of_week' : True,
                'date_featurizer_with_day_of_month' : False,
                'stepwise': True,
                })

            modeling.add_model('holtwinters', {
                # 'trend' : None,
                # 'damped_trend': None,
                # 'seasonal' : None,
                'seasonal_period' : 7,
                'use_boxcox' : False,
                'remove_bias': True,
                })

            modeling.add_model('fbprophet', {})

            modeling.run_modeling(recalculate=True)

        else:
            logger.error(f"Operation type '{operation}' is not supported.")
            sys.exit(1)

        logger.info(f"Total elapsed time {time() - t0:.3f} sec.")
    
    except Exception as err:
        logger.exception(err)

    finally:
        pass
        # sys.exit(1)

if __name__ == '__main__':
    main()