import os
import pickle

import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.serialize import model_to_json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_model(data, holidays):
    # load every column in pd head
    pd.set_option('display.max_columns', None)

    # load the data/transposed first column
    df = pd.read_csv(data)
    print("Data Loaded")

    # keep only the datetime and sales columns
    df = df[["ds", "y"]]
    # datetime column to datetime type
    df["ds"] = pd.to_datetime(df["ds"])

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

    params = {
        'n_changepoints': 50,
        "changepoint_range": 1,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": True,
        'holidays': holidays,
        "seasonality_mode": "multiplicative",
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        "changepoint_prior_scale": 0.01,
        'mcmc_samples': 0,
        'interval_width': 1,
        'uncertainty_samples': 2000,
        'stan_backend': None,
    }

    # fit the model
    print("Training the model")

    model = Prophet(**params).fit(df)  # Fit model with given params
    print("Model training completed")

    # log outputs

    # create results folder
    if not os.path.exists("results"):
        os.makedirs("results")

    folder_name = "train-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

    # inside results folder create folder named  now date(YYYYMMDD)-time(HHMMSS)
    if not os.path.exists("results/" + folder_name):
        os.makedirs("results/" + folder_name)

    # create a file named results.txt
    open("results/" + folder_name + "/results.txt", "w+")

    """LOG"""
    # generate a unique number for each run
    # variable for iteration number
    iteration = str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

    # iteration folder path
    iteration_path = "results/" + folder_name + "/" + iteration
    # inside results/now date(YYYYMMDD)-time(HHMMSS)-iteration folder create folder named iteration number
    if os.path != iteration_path:
        os.makedirs(iteration_path)

    # create results + iteration folder path + .txt file
    open(iteration_path + "/results.txt", "w+")

    # write iteration number, datetime of run to folder_name/results.txt
    with open("results/" + folder_name + "/results.txt", "a") as myfile:
        myfile.write("\n\n============================================================\n")
        myfile.write("Iteration : " + iteration + "\n")
        myfile.write("Datetime : " + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S")) + "\n\n")
    # write params to iteration_path/results.txt
    with open(iteration_path + "/results.txt", "a") as myfile:
        myfile.write("\n\n============================================================\n")
        myfile.write("Iteration : " + iteration + "\n")
        myfile.write("Datetime : " + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S")) + "\n\n")

    # print params
    print("Params: ", params)
    # for each item in params json, write it to the file
    with open("results/" + folder_name + "/results.txt", "a") as results_file:
        results_file.write("Params: \n")
        for key, value in params.items():
            results_file.write("%s: %s\n" % (key, value))
        results_file.write("\n\n")
    with open(iteration_path + "/results.txt", "a") as results_file:
        results_file.write("Params: \n")
        for key, value in params.items():
            results_file.write("%s: %s\n" % (key, value))
        results_file.write("\n\n")

    print("Model Testing...\n")

    future = model.make_future_dataframe(periods=672, freq='15T')
    future.to_csv(iteration_path + "/future-" + iteration + ".csv", index=False)

    # use the model to make a forecast
    forecast = model.predict(future)
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

    # save model to json file in results folder/iteration folder
    with open(iteration_path + '/serialized_model-' + iteration + '.json', 'w') as fout:
        fout.write(model_to_json(model))  # Save model

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
    print("\nModel testing completed. Results saved to results folder.\n")

    pickle.dump(model, open('model.pkl', 'wb'))
    print("Model saved to model.pkl\n\n")

    return df
