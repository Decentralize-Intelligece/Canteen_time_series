import pickle

from sklearn.svm._libsvm import cross_validation
from lib.datetimegen import generate_datetimes as dgt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd


def make_predictions(days,model):
    future = dgt(days, "2020-01-01")
    # save to csv
    future.to_csv("data/future_gen.csv", index=False)

    future = model.make_future_dataframe(periods=200, freq='D')

    future.to_csv("data/future_gen_prophet.csv", index=False)

    print("future")
    print(future.head())

    # use the model to make a forecast
    forecast = model.predict(future)

    # diagnostics
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')



