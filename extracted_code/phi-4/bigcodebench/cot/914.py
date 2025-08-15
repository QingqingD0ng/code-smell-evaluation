import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def task_func(df):
    df['date'] = pd.to_datetime(df['date'])
    df['timestamp'] = df['date'].map(pd.Timestamp.timestamp)
    
    X = df['timestamp'].values.reshape(-1, 1)
    y = df['closing_price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_timestamp = df['timestamp'].iloc[-1]
    future_timestamps = np.array([last_timestamp + i * 24 * 60 * 60 for i in range(1, 8)]).reshape(-1, 1)
    
    future_prices = model.predict(future_timestamps)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['closing_price'], label='Historical Closing Prices')
    future_dates = pd.to_datetime(future_timestamps.flatten())
    plt.plot(future_dates, future_prices, label='Predicted Closing Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Closing Price Prediction')
    plt.legend()
    plt.grid(True)
    
    return future_prices.tolist(), plt.gca()