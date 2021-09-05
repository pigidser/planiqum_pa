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
            logger.debug(f"Initialization of time series data from {data_file}")
            
            # ## Test 1 - daily interval

            # # Load a dataset and define fields purpose.
            # data = Dataset(
            #     filename='./data/orders.csv',
            #     target_col='Quantity',
            #     discrete_interval='day',
            #     date_col='Date',
            #     dimension_col='ProductId')

            # # Create modeling object and define number of intervals for estimation.
            # modeling = Modeling(data, "model_base", n_intervals_estimation=28)

            # # Our dataset includes many products. Let's choose several for test.
            # modeling.define_dimension_values([370, 2354, 2819])
            
            # # Add ARIMA
            # modeling.add_selector(
            #     selector_type='arima',
            #     selector_init_params={'use_boxcox' : True, 'use_date_featurizer': True, 'with_day_of_week' : True, 'with_day_of_month' : False, 'stepwise': True,},
            #     selector_name='test1')

            # # Add Holt-Winter's
            # modeling.add_selector(
            #     selector_type='holtwinters',
            #     selector_init_params={'trend': 'mul', 'damped_trend': False, 'seasonal': 'add', 'seasonal_periods': 7, 'use_boxcox': False, 'remove_bias': True,},
            #     selector_name='test1')

            # # Add Prophet
            # modeling.add_selector(
            #     selector_type='prophet',
            #     selector_init_params={'freq': 'D', 'growth': 'linear', 'yearly_seasonality': False, 'weekly_seasonality': True, 'daily_seasonality': False, 'seasonality_mode': 'additive',},
            #     selector_name='test1')

            # # Select a metric
            # modeling.set_metric('rmse')

            # # Run creation of the best models
            # modeling.start_modeling()

            # # print(modeling.models)

            


            ### Test 2 - weekly interval

            data = Dataset(
                filename='./data/Sales_Product_Price_Store1.csv',
                target_col='Weekly_Units_Sold',
                discrete_interval='week',
                date_col='Date',
                dimension_col='Product')

            modeling = Modeling(data, "model_base", n_intervals_estimation=50)
            modeling.define_dimension_values([1])

            modeling.add_selector(
                selector_type='arima',
                selector_init_params={'seasonal': False, 'use_boxcox': False, 'use_fourier_featurizer': True, 'fourier_featurizer_m': 52, 'fourier_featurizer_k': 4},
                selector_name='with_fourier')

            modeling.add_selector(
                selector_type='arima',
                selector_init_params={'seasonal': True, 'm': 52, 'use_boxcox': False},
                selector_name='native_seasonal')

            modeling.add_selector(
                selector_type='arima',
                selector_init_params={'seasonal': True, 'm': 52, 'use_boxcox': True},
                selector_name='native_seasonal_boxcox')

            # modeling.add_selector(
            #     selector_type='holtwinters',
            #     # selector_init_params={'trend': 'add', 'damped_trend': False, 'seasonal': 'mul', 'use_boxcox': True, 'remove_bias': False},
            #     selector_init_params={'seasonal_periods': 53, 'use_boxcox': True},
            #     selector_name='test2')
            # modeling.add_selector(
            #     selector_type='prophet',
            #     selector_init_params={'growth': 'linear', 'yearly_seasonality': True, 'weekly_seasonality': False, 'daily_seasonality': False, 'seasonality_mode': 'additive', 'freq': 'W'},
            #     selector_name='test2')
            
            modeling.set_metric('smape')
            modeling.start_modeling()

            print(modeling.models)

            # ### Test 3 - monthly interval

            # data = Dataset(
            #     filename='./data/example_retail_sales.csv',
            #     target_col='y',
            #     discrete_interval='month',
            #     date_col='ds',
            #     dimension_col=None)

            # modeling = Modeling(data, "model_base", n_intervals_estimation=28)
            # modeling.add_selector(
            #     selector_type='arima',
            #     selector_init_params={'m': 12},
            #     selector_name='test2')
            # modeling.add_selector(
            #     selector_type='holtwinters',
            #     selector_init_params={'seasonal_periods': 12},
            #     selector_name='test2')
            # modeling.add_selector(
            #     selector_type='prophet',
            #     selector_init_params={'freq': 'MS', 'yearly_seasonality': True, 'seasonality_mode': 'additive'},
            #     selector_name='test2')
            
            # modeling.set_metric('smape')
            # modeling.start_modeling()

        else:
            logger.error(f"Operation type '{operation}' is not supported.")
            sys.exit(1)

        logger.debug(f"Total elapsed time {time() - t0:.3f} sec.")
    
    except Exception as err:
        logger.exception(err)

    finally:
        pass
        # sys.exit(1)

if __name__ == '__main__':
    main()