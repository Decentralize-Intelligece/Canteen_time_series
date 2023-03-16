import pandas as pd
import datetime

def generate_datetimes(x, start_datetime=None):
    if start_datetime is None:
        start_datetime = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        if len(start_datetime) == 10:
            start_datetime += ' 00:00:00'
        start_datetime = datetime.datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        if start_datetime.time() == datetime.time(0, 0, 0):
            start_datetime = start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    datetimes = []
    current_datetime = start_datetime
    for i in range(x):
        for j in range(0, 24 * 4):  # 15 minute intervals for 24 hours
            datetimes.append(current_datetime)
            current_datetime += datetime.timedelta(minutes=15)

    df = pd.DataFrame(datetimes)
    df.columns = ["ds"]
    return df

#
# import datetime
#
# start_datetime = '2023-03-06'  # Example starting date only
# datetimes = generate_datetimes(2, start_datetime)
#
#
#
#
#
# print(df.head())
#
# df.to_csv("data/future_2.csv", index=False)
#
