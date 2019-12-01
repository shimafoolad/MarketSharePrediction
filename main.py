'''
Market share prediction using Linear Regression and Random Forest Regression
'''
import pandas as pd
import argparse

from utils.preprocess import preprocess
from utils.models import models


def Train(df):

    # apply models
    m = models(df, mode='train')

    # Apply linear regression model
    m.LinearRegressionModel()
    LR_model = m.getLRmodel()
    # make predictions and get MAE and R-Squared metrics
    m.Prediction(LR_model)
    print('Linear Regression Model on Validation Data:')
    print("MAE={0:.3f}".format(m.MAE_metric()))
    print("R-Squared={0:.3f}".format(m.RSquared_metric()))

    # Apply Random Forest Regression model
    m.RandomForestModel()
    RF_model = m.getRFmodel()
    # make predictions and get MAE and R-Squared metrics
    m.Prediction(RF_model)
    print('Random Forest Regression Model on Validation Data:')
    print("MAE={0:.3f}".format(m.MAE_metric()))
    print("R-Squared={0:.3f}".format(m.RSquared_metric()))

    return LR_model, RF_model


def Test(df, LR_model, RF_model):


    # apply models
    m = models(df, mode='test')

    # make predictions of linear regression model and get MAE and R-Squared metrics
    m.Prediction(LR_model)
    print('Linear Regression Model on Test Data:')
    print("MAE={0:.3f}".format(m.MAE_metric()))
    print("R-Squared={0:.3f}".format(m.RSquared_metric()))

    # make predictions of random forest regression model and get MAE and R-Squared metrics
    m.Prediction(RF_model)
    print('Random Forest Regression Model on Test Data:')
    print("MAE={0:.3f}".format(m.MAE_metric()))
    print("R-Squared={0:.3f}".format(m.RSquared_metric()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="data/data.csv", help='train file path')
    parser.add_argument('--test_file', type=str, default="data/test.csv", help='test file path')

    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file

    # Import the data-set
    df_train = pd.read_csv(train_file)
    len_train = df_train.__len__()

    df_test = pd.read_csv(test_file)

    df = pd.concat([df_train, df_test])

    # Preprocess the features
    pre = preprocess(df)
    df = pre.getData()

    df_train = df[0:len_train -1]
    df_test = df[len_train:]


    # Train and Test
    LR_model, RF_model = Train(df_train)
    Test(df_test, LR_model, RF_model)
