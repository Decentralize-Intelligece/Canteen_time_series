import json
import csv
import pandas as pd
from datetime import datetime
import os

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
    # Load the JSON data
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    print("Processing data...")
    # Open the CSV file for writing
    with open('data.csv', mode='w') as csv_file:
        # Create a CSV writer object
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(['date', 'sales', 'time', 'weekday', 'holiday'])

        # Write each row of data
        for date, values in data.items():
            writer.writerow([date, values['sales'], values['time'], values['weekday'], values['holiday']])

        df = pd.read_csv('data.csv')

        # keep only the datetime and sales columns
        df = df[["date", "sales"]]

        # datetime column to datetime type
        df["date"] = pd.to_datetime(df["date"])

        # rename the columns
        df.columns = ["ds", "y"]

        # Reset the index of the DataFrame
        df = df.reset_index(drop=True)

        # Sort the DataFrame by the 'date' column
        df = df.sort_values(by='ds')

        # *******************

        # Set 'datetime' as the index of the DataFrame
        df = df.set_index('ds')

        # Create a new DataFrame with a datetime range that spans the entire time period
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15T')
        new_df = pd.DataFrame(index=date_range)

        # Join the new DataFrame with the original DataFrame on the datetime index
        df = new_df.join(df, how='left')

        # Fill missing sales values with zeros
        df['y'] = df['y'].fillna(0)

        # Resample the DataFrame with a 15-min frequency and fill missing values with zero sales
        # df = df.resample('15T').fillna(0)

        # Reset the index of the DataFrame
        df = df.reset_index()

        # create results folder
        if not os.path.exists("data"):
            os.makedirs("data")

        folder_name = "data-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

        # create a folder for the results
        if not os.path.exists("data/" + folder_name):
            os.mkdir("data/" + folder_name)

        # save the dataframe as a csv file
        df.to_csv("data/" + folder_name + "/processed_data.csv", index=False)

        bp_indices = find_csv_breaking_points("data/" + folder_name + "/processed_data.csv", datetime_format, threshold)
        if len(bp_indices) == 0:
            print("The dataset is continuous")
        else:
            print("The dataset has", len(bp_indices), "breaking points at indices:", bp_indices)

        return folder_name + "/processed_data.csv"

