import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def task_func(df):
    df.index = pd.to_datetime(df['date'])
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['closing_price'].values
    model = LinearRegression()
    model.fit(X, y)
    future_steps = np.array(range(len(df), len(df) + 7)).reshape(-1, 1)
    future_prices = model.predict(future_steps)
    
    fig, ax = plt.subplots()
    ax.scatter(df.index, y, color='blue')
    ax.plot(df.index, model.predict(X), color='green')
    ax.plot(range(len(df), len(df) + 7), future_prices, color='red', linestyle='dashed')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.legend(['Trend Line', 'Predicted Trend Line', 'Predicted Prices'])
    
    return list(future_prices), ax