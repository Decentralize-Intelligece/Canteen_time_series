import numpy as np
import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def warm_start_params(m):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res


df_original_data = pd.read_csv("data/df3.csv")
df_retrain_data = pd.read_csv("data/df2.csv")

m1 = Prophet().fit(df_original_data)  # Fitting from scratch

# diagnostics
df_cv = cross_validation(m1, initial='730 days', period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)

print("df_cv")
print(df_cv.head())
print("df_p")
print(df_p.head())

# you can load the model from json
# m1 = model_from_json(model_to_json(m1))

print("Model 1 trained")

m2 = Prophet().fit(df_retrain_data, init=warm_start_params(m1))  # Adding the last day, warm-starting from m1

# forecast
future = m2.make_future_dataframe(periods=12)
forecast = m2.predict(future)

df = df_original_data.append(df_retrain_data)

# diagnostics
df_cv = cross_validation(m2, initial='130 days', period='45 days', horizon='65 days')
df_p = performance_metrics(df_cv)

print("df_cv")
print(df_cv.head())
print("df_p")
print(df_p.head())

# plot forecast
m2.plot(forecast)
pyplot.show()

# plot forecast components
m2.plot_components(forecast)
pyplot.show()
