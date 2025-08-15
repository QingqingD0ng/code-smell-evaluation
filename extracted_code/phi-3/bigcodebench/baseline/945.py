import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.frequencies import to_offset

def generate_business_dates(start_date, periods, freq):
    """
    Generates a series of business dates based on the specified frequency.
    """
    if freq == 'WOM-2FRI':
        offset = CustomBusinessDay(weekday=4)  # Friday
        start_date = pd.to_datetime(start_date)
        business_date = start_date
        date_range = []
        for _ in range(periods):
            business_date += offset
            date_range.append(business_date)
        return np.array(date_range)
    else:
        raise ValueError("Unsupported frequency. Use 'WOM-2FRI'.")

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    if sales_data is None:
        np.random.seed(42)
        sales_data = np.random.rand(periods) * 100
    dates = generate_business_dates(start_date, periods, freq)
    df = pd.DataFrame({'date': dates,'sales': sales_data})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    X = np.array(range(periods)).reshape(-1, 1)
    y = df['sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_periods = np.array(range(periods, periods + 13)).reshape(-1, 1)
    forecast = model.predict(future_periods)
    
    return forecast

# Examples
if __name__ == "__main__":
    np.random.seed(42)
    print