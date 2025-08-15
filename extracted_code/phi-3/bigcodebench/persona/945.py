import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from pandas.tseries.offsets import MonthEnd


def generate_sales_data(start_date, periods, freq):
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    date_range = date_range.append(date_range + MonthEnd(1)) - MonthEnd(1)
    dates = date_range.strftime('%Y-%m-%d').tolist()
    sales = np.random.randint(low=200, high=800, size=len(dates)).tolist()
    return np.array(dates), np.array(sales)


def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    if sales_data is None:
        dates, sales = generate_sales_data(start_date, periods, freq)
    else:
        dates, sales = sales_data
        
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_periods = np.array(range(len(df), len(df) + periods)).reshape(-1, 1)
    forecast = model.predict(future_periods)
    
    return forecast