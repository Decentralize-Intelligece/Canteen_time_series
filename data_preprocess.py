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

        # save the dataframe as a csv file
        df.to_csv('processed_data.csv', index=False)

        print(df.head(1000))


if __name__ == "__main__":
    json_file_path = input("Enter the path to the JSON file: ")
    csv_file_path = json_to_csv(json_file_path)


