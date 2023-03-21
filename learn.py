import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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


def learn(old_data, new_data, holidays, model):
    df_original_data = pd.read_csv(old_data)
    df_retrain_data = pd.read_csv(new_data)

    # load holidays
    holidays = pd.read_csv(holidays)
    print("Holidays Loaded")

    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        # holidays as 'ds'
        'ds': holidays["ds"],
        'lower_window': 0,
        'upper_window': 1,
    })

    holidays = playoffs

    print("Previous model loaded")
    m2 = Prophet(
        holidays=holidays,
        n_changepoints=50,
        changepoint_prior_scale=0.01,
        changepoint_range=1,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0, interval_width=1,
        uncertainty_samples=2000,
        stan_backend=None
    ).fit(df_retrain_data, init=warm_start_params(model))  # Adding the last day, warm-starting from model 1

    future = m2.make_future_dataframe(periods=12)
    forecast = m2.predict(future)

    df = df_original_data.append(df_retrain_data)

    pickle.dump(m2, open('new_model.pkl', 'wb'))
    print("New model saved as 'new_model.pkl ")

    # create results folder
    if not os.path.exists("results"):
        os.makedirs("results")

    # log files
    iteration = str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))
    folder_name = "learn-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

    # iteration folder path
    iteration_path = "results/" + folder_name + "/" + iteration
    # inside results/now date(YYYYMMDD)-time(HHMMSS)-iteration folder create folder named iteration number
    if not os.path.exists(iteration_path):
        os.makedirs(iteration_path)

    forecast.to_csv(iteration_path + "/forecast-" + iteration + ".csv", index=False)

    r2_score(df["y"], forecast["yhat"][:len(df["y"])])
    mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])
    mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])

    print("Metrics")
    print("R2: ", r2_score(df["y"], forecast["yhat"][:len(df["y"])]))
    print("MSE: ", mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])]))
    print("MAE: ", mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])]))

    # Metrics to log
    with open("results/" + folder_name + "/results.txt", "a") as myfile:
        myfile.write("\n\nMetrics\n")
        myfile.write("R2: ")
        myfile.write(str(r2_score(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\nMSE: ")
        myfile.write(str(mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\nMAE: ")
        myfile.write(str(mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\n\n")

    # plot forecast
    m2.plot(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast-" + iteration + ".png")

    # plot forecast components
    m2.plot_components(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast_components-" + iteration + ".png")

    # plot changepoints
    fig = m2.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m2, forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/changepoints-" + iteration + ".png")
