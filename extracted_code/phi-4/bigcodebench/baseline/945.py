import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    if sales_data is None:
        np.random.seed(42)  # For consistent random data generation in examples
        sales_data = np.random.randint(100, 500, size=periods)

    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    df = pd.DataFrame({'Date': date_range, 'Sales': sales_data})

    df['TimeIndex'] = np.arange(len(df))
    X = df[['TimeIndex']]
    y = df['Sales']

    model = LinearRegression()
    model.fit(X, y)

    future_periods = np.arange(len(df), len(df) + periods).reshape(-1, 1)
    forecast = model.predict(future_periods)

    return forecast