import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    # Generate date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate or use provided sales data
    if sales_data is None:
        np.random.seed(42)
        sales_data = np.random.randint(100, 500, size=periods)
    
    # Create DataFrame
    df = pd.DataFrame({'date': dates,'sales': sales_data})
    
    # Prepare data for linear regression
    df['time'] = np.arange(len(df))
    X = df[['time']]
    y = df['sales']
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future sales
    future_time = np.arange(len(df), len(df) * 2).reshape(-1, 1)
    forecasted_sales = model.predict(future_time)
    
    return forecasted_sales