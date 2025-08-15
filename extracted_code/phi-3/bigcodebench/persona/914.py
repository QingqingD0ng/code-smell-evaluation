import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta


def task_func(df):
    df['timestamp'] = (df['date'] - df['date'].min()).dt.total_seconds()
    X = df['timestamp'].values.reshape(-1, 1)
    y = df['closing_price'].values
    model = LinearRegression()
    model.fit(X, y)
    future_timestamps = np.array([df['timestamp'].iloc[-1] + i*24*60*60 for i in range(1, 8)]).reshape(-1, 1)
    future_prices = model.predict(future_timestamps)
    plt.scatter(df['timestamp'], df['closing_price'], color='blue')
    plt.plot(future_timestamps, future_prices, color='red')
    plt.xlabel('Timestamp (seconds since start)')
    plt.ylabel('Closing Price')
    ax = plt.gca()
    ax.set_xticks(future_timestamps)
    plt.xticks(rotation=45)
    return future_prices.tolist(), ax


# Example usage:

df = pd.DataFrame({
    'date': pd.date_range(start='1/1/2021', periods=7),
    'closing_price': [100, 101, 102, 103, 104, 105, 106]
})

pred_prices, plot = task_func(df)
print(pred_prices)