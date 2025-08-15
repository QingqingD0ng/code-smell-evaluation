import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    if sales_data is None:
        sales_data = np.random.rand(periods) * 1000
    
    df = pd.DataFrame({'Date': date_range, 'Sales': sales_data})
    df['TimeIndex'] = np.arange(len(df))
    
    X = df[['TimeIndex']]
    y = df['Sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_time_index = np.arange(len(df), len(df) * 2).reshape(-1, 1)
    future_sales = model.predict(future_time_index)
    
    return future_sales