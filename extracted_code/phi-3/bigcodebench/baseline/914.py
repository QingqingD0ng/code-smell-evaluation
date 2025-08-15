import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import timedelta


def task_func(df):

    df['timestamp'] = (df['date'] - df['date'].min()).dt.total_seconds()

    X = df['timestamp'].values.reshape(-1, 1)

    y = df['closing_price'].values

    model = LinearRegression()

    model.fit(X, y)

    last_date = df['date'].iloc[-1]

    future_days = np.array([(last_date + timedelta(days=i)) for i in range(1, 8)]).reshape(-1, 1)

    future_timestamps = (future_days - df['date'].min()).dt.total_seconds().reshape(-1, 1)

    future_prices = model.predict(future_timestamps)


    plt.scatter(df['timestamp'], y, color='blue')

    plt.plot(X, model.predict(X), color='red')

    plt.scatter(future_timestamps, future_prices, color='green')

    plt.xlabel('Timestamp')

    plt.ylabel('Closing Price')

    plt.title('Stock Closing Price Prediction')

    plt.legend(['Train', 'Future Prediction'])

    fig = plt.gcf()

    axes = fig.get_axes()


    return list(future_prices), axes