"""Train model and save checkpoint"""

import argparse
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='C:/Users/232556/PycharmProjects/pabd24/log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'C:/Users/232556/PycharmProjects/pabd24/data/proc/train.csv'
VAL_DATA = 'C:/Users/232556/PycharmProjects/pabd24/data/proc/val.csv'
MODEL_SAVE_PATH = 'C:/Users/232556/PycharmProjects/pabd24/models/linear_regression_v01.joblib'


def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    x_train = df_train[['total_meters']]
    y_train = df_train['price']

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    dump(linear_model, args.model)
    logger.info(f'Saved to {args.model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)