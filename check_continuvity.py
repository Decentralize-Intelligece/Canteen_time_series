import os
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path

filename = "D:\Git Hub Projects\Time Series Project\Project\process_data\processed_data.csv"
datetime_format = '%Y-%m-%d %H:%M:%S'
threshold = 900 # in seconds


def find_csv_breaking_points(filename, datetime_format, threshold):
    breaking_points = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader) # read the header row
        data = sorted(reader, key=lambda x: datetime.strptime(x[0], datetime_format)) # sort the data by datetime column

        prev_time = None
        for i, row in enumerate(data):
            current_time = datetime.strptime(row[0], datetime_format)
            if prev_time is not None:
                if abs(((current_time - prev_time).total_seconds())) > threshold:
                    breaking_points.append(i)
            prev_time = current_time
    return breaking_points


bp_indices = find_csv_breaking_points(filename, datetime_format, threshold)
if len(bp_indices) == 0:
    print("The dataset is continuous")
else:
    print("The dataset has", len(bp_indices), "breaking points at indices:", bp_indices)
