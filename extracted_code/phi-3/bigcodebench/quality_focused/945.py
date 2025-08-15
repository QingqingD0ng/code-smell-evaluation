import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    # Generate dates for the time series
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Initialize sales data with random values if not provided
    if sales_data is None:
        np.random.seed(42)
        sales_data = np.random.rand(periods) * 400 + 200
    
    # Prepare the data for linear regression
    X = np.arange(len(dates)).reshape(-1, 1)
    y = sales_data
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future sales for the same number of periods
    future_X = np.arange(len(dates), len(dates) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    return forecast