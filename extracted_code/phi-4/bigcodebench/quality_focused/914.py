import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def task_func(df):
    # Ensure 'date' is in datetime format and sort the dataframe by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Convert dates to ordinal numbers
    df['ordinal_date'] = df['date'].map(pd.Timestamp.toordinal)
    
    # Prepare data for linear regression
    X = df[['ordinal_date']]
    y = df['closing_price']
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = df['date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    future_ordinal_dates = [d.toordinal() for d in future_dates]
    
    # Predict future closing prices
    future_X = np.array(future_ordinal_dates).reshape(-1, 1)
    pred_prices = model.predict(future_X).tolist()
    
    # Plot the data
    plt.figure()
    plt.plot(df['date'], df['closing_price'], label='Actual Prices')
    plt.plot(future_dates, pred_prices, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Closing Prices Prediction')
    plt.legend()
    ax = plt.gca()
    
    return pred_prices, ax