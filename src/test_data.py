"""test model and save checkpoint"""

import argparse
import logging
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from joblib import load

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

    df_val = pd.read_csv(VAL_DATA)
    x_val = df_val[['total_meters']]
    y_val = df_val['price']

    linear_model = load('C:/Users/232556/PycharmProjects/pabd24/models/linear_regression_v01.joblib')

    r2 = linear_model.score(x_train, y_train)
    y_pred = linear_model.predict(x_val)
    mae = mean_absolute_error(y_pred, y_val)
    c = int(linear_model.coef_[0])
    inter = int(linear_model.intercept_)

    logger.info(f'R2 = {r2:.3f}     MAE = {mae:.0f}     Price = {c} * area + {inter}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)