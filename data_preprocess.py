import json
import csv
import pandas as pd


def json_to_csv(json_file_path):
    import json
    import csv

    # Load the JSON data
    with open(json_file_path) as json_file:
        data = json.load(json_file)

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

        #*******************

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

        # *********************

        # save the dataframe as a csv file
        df.to_csv('processed_data.csv', index=False)

        print(df.head(1000))


if __name__ == "__main__":
    json_file_path = input("Enter the path to the JSON file: ")
    csv_file_path = json_to_csv(json_file_path)

# D:\Git Hub Projects\Time Series Project\Project\data\unprocessed_data\interval_counts.json


