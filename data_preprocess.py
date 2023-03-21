import csv
import json
import os
from datetime import datetime

import pandas as pd

datetime_format = '%Y-%m-%d %H:%M:%S'
threshold = 900  # in seconds


def find_csv_breaking_points(filename, datetime_format, threshold):
    breaking_points = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # read the header row
        data = sorted(reader,
                      key=lambda x: datetime.strptime(x[0], datetime_format))  # sort the data by datetime column

        prev_time = None
        for i, row in enumerate(data):
            current_time = datetime.strptime(row[0], datetime_format)
            if prev_time is not None:
                if abs(((current_time - prev_time).total_seconds())) > threshold:
                    breaking_points.append(i)
            prev_time = current_time
    return breaking_points


def json_to_csv(json_file_path):
    print("Converting JSON to CSV...")
    # Load the JSON data to dataframes
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    print("Processing data...")
    # data to dataframe
    df = pd.DataFrame(data)
    # transpose the dataframe
    df = df.transpose()

    # save to csv
    df.to_csv("data.csv", header=False)

    # read the csv file
    df = pd.read_csv("data.csv", header=None)

    # rename the columns
    df = df.rename(columns={0: 'ds', 1: 'y', 2: 'time', 3: 'weekday', 4: 'holiday'})

    # keep only first two columns
    df = df.iloc[:, :2]

    # ds to datetime type
    df["ds"] = pd.to_datetime(df["ds"])

    print("Sorting data...")
    # sort the dataframe by ds
    df = df.sort_values(by=['ds'])

    # reset the index
    df = df.reset_index(drop=True)

    # Create a new DataFrame with a datetime range that spans the entire time period
    # df start date
    start_date = df['ds'].min()
    end_date = df['ds'].max()

    # create a new dataframe from start date to end date with 15 minutes interval
    new_df = pd.DataFrame(pd.date_range(start_date, end_date, freq='15min'), columns=['ds'])

    # Fill new_df values from df
    new_df = new_df.merge(df, on='ds', how='left')

    # Fill missing sales values with zeros no decimal
    new_df['y'] = new_df['y'].fillna(0).astype(int)

    # save the dataframe as a csv file
    new_df.to_csv("Filled.csv", index=False)

    # create results folder
    if not os.path.exists("data"):
        os.makedirs("data")

    folder_name = "data-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

    # create a folder for the results
    if not os.path.exists("data/" + folder_name):
        os.mkdir("data/" + folder_name)

    # save the dataframe as a csv file
    new_df.to_csv("data/" + folder_name + "/processed_data.csv", index=False)
    new_df.to_csv("data/processed_data.csv", index=False)

    bp_indices = find_csv_breaking_points("data/" + folder_name + "/processed_data.csv", datetime_format, threshold)
    if len(bp_indices) == 0:
        print("\nThe dataset is continuous")
    else:
        print("\nThe dataset has", len(bp_indices), "breaking points at indices:", bp_indices)

    return folder_name + "/processed_data.csv"
