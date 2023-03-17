import pandas as pd
from prophet import Prophet


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

    # print(df.dtypes)
    # print(df.head())

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

    # add holidays
    model = Prophet(
        holidays=holidays,
        changepoint_prior_scale=1, changepoint_range=1,
        yearly_seasonality=True, weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=500.0, holidays_prior_scale=500.0,
        mcmc_samples=0, interval_width=1,
        uncertainty_samples=2000, stan_backend=None,
        # growth='logistic'
    )

    # add regressors
    # model.add_regressor("y")

    # Metrics
    # R2:  0.383919577279893
    # MSE:  2427.8544877382374
    # MAE:  17.901745566393274

    # R2:  0.4085147857617024
    # MSE:  2330.9295002085705
    # MAE:  18.468722578249313

    # fit the model
    print("Training the model")

    model.fit(df)
    print("Model training completed")
