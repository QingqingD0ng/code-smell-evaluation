import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_time_series(start_date, periods, freq):
    date_range = pd.date_range(start_date, periods=periods, freq=freq)
    sales_data = np.random.rand(len(date_range)) * 1000
    return sales_data

def forecast_sales(start_date, periods, freq, sales_data=None):
    if sales_data is None:
        sales_data = generate_time_series(start_date, periods, freq)
    X = np.arange(len(sales_data)).reshape(-1, 1)
    y = sales_data
    model = LinearRegression()
    model.fit(X, y)
    future_periods = np.arange(len(sales_data), len(sales_data) + periods).reshape(-1, 1)
    forecasted_sales = model.predict(future_periods)
    return forecasted_sales

# Example usage:
# np.random.seed(42)
# forecast_sales('2016-01-01', 13, 'WOM-2FRI')
# forecast_sales('2020-01-01', 5, 'M', [200, 300, 400, 500, 600])