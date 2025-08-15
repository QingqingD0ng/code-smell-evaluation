import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    if sales_data is None:
        sales_data = np.random.rand(periods) * 1000  # Generate random sales data
    sales_series = pd.Series(sales_data, index=dates)
    
    X = np.arange(len(sales_series)).reshape(-1, 1)
    y = sales_series.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_periods = np.arange(len(sales_series), 2 * len(sales_series)).reshape(-1, 1)
    forecast = model.predict(future_periods)
    
    return forecast