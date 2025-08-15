import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def task_func(df):
    # Convert dates to ordinal
    df['ordinal_date'] = df['date'].apply(lambda x: x.toordinal())
    
    # Prepare the data for linear regression
    X = df['ordinal_date'].values.reshape(-1, 1)
    y = df['closing_price'].values
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the next 7 days
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predicted_prices = model.predict(future_ordinals)
    
    # Plot the data
    plt.figure()
    plt.plot(df['date'], df['closing_price'], label='Actual Prices')
    plt.plot(future_dates, predicted_prices, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Closing Prices')
    plt.legend()
    ax = plt.gca()
    
    return predicted_prices.tolist(), ax