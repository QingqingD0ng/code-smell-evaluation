import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def task_func(df):
    df = df.sort_values('date')
    df['timestamp'] = df['date'].astype(np.int64) // 10**9
    X = df[['timestamp']]
    y = df['closing_price']
    model = LinearRegression()
    model.fit(X, y)

    last_date = df['date'].iat[-1]
    future_dates = [last_date + timedelta(seconds=i*3600*24) for i in range(1, 8)]
    future_timestamps = np.array([(date - df['date'].min()).total_seconds() for date in future_dates]).reshape(-1, 1)
    future_prices = model.predict(future_timestamps)

    plt.scatter(df['date'], df['closing_price'], color='blue')
    plt.plot(df['date'], model.predict(X), color='red')
    plt.plot(future_dates, future_prices, color='green')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Price Prediction')
    plt.show()

    return list(future_prices), plt