import os

import pandas as pd
from matplotlib import pyplot
from prophet.plot import add_changepoints_to_plot


def make_predictions(days, model, num_of_days):
    # find the last date by subtracting num_of_days from today
    last_date = (pd.to_datetime('today') - pd.Timedelta(days=num_of_days)).strftime("%Y-%m-%d")
    print("\nMaking predictions for the next " + str(days) + " days from " + str(last_date.format()))

    future = model.make_future_dataframe(periods=24 * 4 * (days + num_of_days), freq='15T')
    future.to_csv("data/future_gen_prophet.csv", index=False)

    # from future copy to df_future the last 24*4*days + last_date rows
    df_future = future[len(future) - 24 * 4 * (days + num_of_days):len(future)]
    df_future.to_csv("data/future_gen.csv", index=False)

    print("Forecasting...")
    # use the model to make a forecast
    forecast = model.predict(future)

    # create results folder
    if not os.path.exists("results"):
        os.makedirs("results")

    """log files"""
    iteration = str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))
    folder_name = "predict-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

    # iteration folder path
    iteration_path = "results/" + folder_name + "/" + iteration
    # inside results/now date(YYYYMMDD)-time(HHMMSS)-iteration folder create folder named iteration number
    if not os.path.exists(iteration_path):
        os.makedirs(iteration_path)

    forecast.to_csv(iteration_path + "/forecast-" + iteration + ".csv", index=False)

    # plot forecast
    model.plot(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast-" + iteration + ".png")

    # plot forecast components
    model.plot_components(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast_components-" + iteration + ".png")

    # plot changepoints
    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/changepoints-" + iteration + ".png")

    print("Predictions saved to " + iteration_path)
