import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm._libsvm import cross_validation

from lib.datetimegen import generate_datetimes as dgt

isNotValid = True
days = 0

pd.set_option('display.max_columns', None)

# load the data/transposed first column
df = pd.read_csv("data/df3.csv")

# keep only the datetime and sales columns
df = df[["ds", "y"]]
# datetime column to datetime type
df["ds"] = pd.to_datetime(df["ds"])

while (isNotValid):
    try:
        print('Enter for how many days you need the forecast :')
        days = int(input())
        isNotValid = False

    except:
        print('Enter a valid input....')

# load model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))

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


def performance_metrics(df_cv):
    pass


df_p = performance_metrics(df_cv)

print("df_cv")
print(df_cv.head())
print("df_p")
print(df_p.head())

# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

r2_score(df["y"], forecast["yhat"][:len(df["y"])])
mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])
mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])

print("Metrics")
print("R2: ", r2_score(df["y"], forecast["yhat"][:len(df["y"])]))
print("MSE: ", mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])]))
print("MAE: ", mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])]))
