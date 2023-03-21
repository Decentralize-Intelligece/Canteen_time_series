import pickle

import pandas as pd

import learn
import predict
import train
from data_preprocess import json_to_csv


def train_console():
    print("\n\n___________________________________________________________")
    print("Training the model")
    print("___________________________________________________________")
    print("TODO : Copy your data file and holidays file to data folder")
    print("___________________________________________________________")
    data = input("Enter your data file name (relative) : ")
    holidays = input("Enter your holidays file name (relative) : ")

    data = "data/" + data
    holidays = "data/" + holidays

    df = train.train_model(data, holidays)

    return df


def predict_console():
    print("\n\n___________________________________________________________")
    print("Ready to make predictions")
    print("___________________________________________________________")
    print("TODO : Copy your data files to data folder")
    data = input("Enter the name of the data file that was used to train the model (relative) : ")

    path = "data/" + data

    pd.set_option('display.max_columns', None)

    # load the data/transposed first column
    df = pd.read_csv(path)
    print("Data Loaded")

    # keep only the datetime and sales columns
    df = df[["ds", "y"]]
    # datetime column to datetime type
    df["ds"] = pd.to_datetime(df["ds"])

    # get the last datetime
    last_date = df["ds"].iloc[-1]

    print("Last date in the dataset is : " + str(last_date))
    # calculate days from last date to today
    num_of_days = (pd.to_datetime('today') - last_date).days
    if num_of_days > 0:
        print("Number of days from last date to today is : " + str(num_of_days))
        print("Recommend to re-train/learn the model with new data if you need to have good forecast from today")
    elif num_of_days < 0:
        print("Your data set had future dates of : " + str(num_of_days) + " days")

    isNotValid = True
    days = 0

    while (isNotValid):
        try:
            print('\nEnter for how many days you need the forecast from last date: ', end="")
            days = int(input())
            isNotValid = False

        except:
            print('Enter a valid input....')

    model = pickle.load(open('model.pkl', 'rb'))

    predict.make_predictions(days, model, num_of_days)


def learn_console():
    print("\n\n___________________________________________________________")
    print("Re-train the model with new data")
    print("___________________________________________________________")
    print("TODO : Copy your data files to data folder")
    print("___________________________________________________________")
    old_data = input("Enter your old data file name (relative) : ")
    new_data = input("Enter your new data file name (relative) : ")
    holidays = input("Enter your holidays file name (relative) : ")

    old_data = "data/" + old_data
    new_data = "data/" + new_data
    holidays = "data/" + holidays

    model = pickle.load(open('model.pkl', 'rb'))

    learn.learn(old_data, new_data, holidays, model)


def process_console():
    print("\n\n___________________________________________________________")
    print("Ready to process data")
    print("___________________________________________________________")
    print("TODO : Enter the path to the JSON file:", end=" ")
    json_file_path = input()
    csv_file_path = json_to_csv(json_file_path)

    print(
        "\nDone! The processed data is saved in the folder:" + csv_file_path + " and to " + "data/processed_data.csv\n")


def switch_console(option):
    if option == 1:
        process_console()
    elif option == 2:
        train_console()
    elif option == 3:
        predict_console()
    elif option == 4:
        learn_console()
    elif option == 5:
        exit_console()
    else:
        print("Not a valid option")


def exit_console():
    print("___________________________________________________________")
    print("Exiting the program")
    print("___________________________________________________________")

    exit()


def welcome_screen():
    print("Time Series Solution")
    print("Options : ")
    print("     1 - preprocess data")
    print("     2 - train the model")
    print("     3 - predict")
    print("     4 - learn")
    print("     5 - exit")
    option = int(input("What do you want to do (enter the number) :"))
    switch_console(option)
