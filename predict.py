import os
import pickle

from matplotlib import pyplot
from sklearn.svm._libsvm import cross_validation
from lib.datetimegen import generate_datetimes as dgt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import train
import pandas as pd


def make_predictions(days,model,df):
    future = dgt(days, "2020-01-01")
    # save to csv
    future.to_csv("data/future_gen.csv", index=False)


    future = model.make_future_dataframe(periods=24*4*days, freq='D')

    future.to_csv("data/future_gen_prophet.csv", index=False)

    print("future")
    print(future.head())

    # use the model to make a forecast
    forecast = model.predict(future)





    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

    r2_score(df["y"], forecast["yhat"][:len(df["y"])])
    mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])
    mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])

    print("Metrics")
    print("R2: ", r2_score(df["y"], forecast["yhat"][:len(df["y"])]))
    print("MSE: ", mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])]))
    print("MAE: ", mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])]))

    # plot forecast
    model.plot(forecast)
    pyplot.show()

    # plot forecast components
    model.plot_components(forecast)
    pyplot.show()

    from prophet.plot import add_changepoints_to_plot

    fig = model.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), model, forecast)
    pyplot.show()

    #log files

    iteration = str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))
    folder_name = "prophet-single-run-results-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

    # iteration folder path
    iteration_path = "results/" + folder_name + "/" + iteration
    # inside results/now date(YYYYMMDD)-time(HHMMSS)-iteration folder create folder named iteration number
    if os.path != iteration_path:
        os.makedirs(iteration_path)
    forecast.to_csv(iteration_path + "/forecast-" + iteration + ".csv", index=False)

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
    model.plot(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast-" + iteration + ".png")

    # plot forecast components
    model.plot_components(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast_components-" + iteration + ".png")

    # plot changepoints
    fig = model.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), model, forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/changepoints-" + iteration + ".png")











