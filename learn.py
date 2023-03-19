import pickle

import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np


def warm_start_params(m):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res


def learn(old_data, new_data):
    df_original_data = pd.read_csv(old_data)
    df_retrain_data = pd.read_csv(new_data)

    m1 = Prophet().fit(df_original_data)  # Fitting from scratch

    # diagnostics
    df_cv = cross_validation(m1, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)

    # print("df_cv")
    # print(df_cv.head())
    # print("df_p")
    # print(df_p.head())

    # m1 = pickle.load(open('model.pkl', 'rb'))
    print("Previous model loaded")
    m2 = Prophet().fit(df_retrain_data, init=warm_start_params(m1))  # Adding the last day, warm-starting from m1

    future = m2.make_future_dataframe(periods=12)
    forecast = m2.predict(future)

    df = df_original_data.append(df_retrain_data)

    # diagnostics
    df_cv = cross_validation(m2, initial='130 days', period='45 days', horizon='65 days')
    df_p = performance_metrics(df_cv)

    # print("df_cv")
    # print(df_cv.head())
    # print("df_p")
    # print(df_p.head())

    # plot forecast
    m2.plot(forecast)
    pyplot.show()

    # plot forecast components
    m2.plot_components(forecast)
    pyplot.show()

    pickle.dump(m2, open('new_model.pkl', 'wb'))
    print("New model saved as 'new_model.pkl ")
