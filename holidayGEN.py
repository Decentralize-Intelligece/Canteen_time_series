# 2018-04-01 10:00:00 - 2022-12-31 23:30:00
#
# Holidays for CanteenW:
# 07.01.2023-11.02.2023
# 24.10.2022-24.12.2022
# 04.04.2022-09.07.2022
# 05.01.2022-05.02.2022
# 18.10.2021-20.12.2021
# 12.04.2021-17.07.2021
# 02.01.2021-06.02.2021
# 19.10.2020-19.12.2020
# 20.04.2020-18.07.2020
import pandas as pd


def holiday_gen(start_date, end_date, data_range_start="2018-04-01 10:00:00", data_range_end="2023-03-31 23:30:00"):
    """
    Generate holidays for the given date range
    :param start_date:
    :param end_date:
    :return:
    """

    # convert start_date to datetime
    start_date = pd.to_datetime(start_date)
    # convert end_date to datetime
    end_date = pd.to_datetime(end_date)
    # convert data_range_start to datetime
    data_range_start = pd.to_datetime(data_range_start)
    # convert data_range_end to datetime
    data_range_end = pd.to_datetime(data_range_end)

    # if start_date is before data_range_start
    if start_date < data_range_start:
        # set start_date to data_range_start
        start_date = data_range_start
    # if end_date is after data_range_end
    if end_date > data_range_end:
        # set end_date to data_range_end
        end_date = data_range_end


    # create dataframe from start_date to end_date in 15 minute intervals
    df = pd.DataFrame()
    df['ds'] = pd.date_range(start_date, end_date, freq='15T')


    return df


dates = [
    "07.01.2023-11.02.2023",
    "24.10.2022-24.12.2022",
    "04.04.2022-09.07.2022",
    "05.01.2022-05.02.2022",
    "18.10.2021-20.12.2021",
    "12.04.2021-17.07.2021",
    "02.01.2021-06.02.2021",
    "19.10.2020-19.12.2020",
    "20.04.2020-18.07.2020",
]

def gen_holidays(dates):
    """
    Generate holidays for the given dates
    :param dates:
    :return:
    """
    # create a new dataframe
    df = pd.DataFrame()
    # for each date
    for date in dates:
        # split the date by -
        date = date.split("-")
        # split the first date by .
        date1 = date[0].split(".")
        # split the second date by .
        date2 = date[1].split(".")
        # create a new dataframe from the first date to the second date
        df2 = holiday_gen(
            "{}-{}-{} 00:00:00".format(date1[2], date1[1], date1[0]),
            "{}-{}-{} 23:59:59".format(date2[2], date2[1], date2[0])
        )
        # append the new dataframe to the old dataframe
        df = df.append(df2)

    # sort the dataframe by ds
    df = df.sort_values(by="ds")

    # return the dataframe
    return df

# generate holidays
df = gen_holidays(dates)
# print the holidays
print(df)

# save to csv
df.to_csv("data/holidays-2.csv", index=False)
