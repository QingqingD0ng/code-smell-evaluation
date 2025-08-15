import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta


# Constants

TIME_STEP = timedelta(days=1).total_seconds()

NUM_DAYS_TO_PREDICT = 7


def task_func(df, prediction_days=NUM_DAYS_TO_PREDICT):

    if 'date' not in df.columns or 'closing_price' not in df.columns:

        raise ValueError("DataFrame must contain 'date' and 'closing_price' columns")


    df['timestamp'] = (df['date'] - df['date'].min()).dt.total_seconds()

    X = df['timestamp'].values.reshape(-1, 1)

    y = df['closing_price'].values

    model = LinearRegression()

    model.fit(X, y)

    last_date = df['date'].iloc[-1]

    future_dates = [last_date + timedelta(days=i) for i in range(prediction_days)]

    future_timestamps = [(future_date - df['date'].min()).total_seconds() for future_date in future_dates]

    future_timestamps = np.array(future_timestamps).reshape(-1, 1)

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


    return list(future_pr