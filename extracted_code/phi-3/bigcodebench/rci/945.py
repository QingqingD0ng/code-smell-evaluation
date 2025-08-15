import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import CustomBusinessDay

def generate_business_dates(start_date, periods, freq):
    offset = CustomBusinessDay(weekday=4)  # Friday
    start_date = pd.to_datetime(start_date)
    business_dates = [start_date + i * offset for i in range(periods)]
    return np.array(business_dates)

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', sales_data=None):
    if sales_data is None:
        sales_data = np.random.rand(periods) * 100
    dates = generate_business_dates(start_date, periods, freq)
    df = pd.DataFrame({'date': dates,'sales': sales_data})
    df.set_index('date', inplace=True)
    
    X = np.array(range(periods)).reshape(-1, 1)
    y = df['sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_periods = np.array(range(periods, periods + 13)).reshape(-1, 1)
    forecast = model.predict(future_periods)
    
    return forecast

if __name__ == "__main__":
    np.random.seed(42)
    print(task_func('2016-01-01', 13, 'WOM-2FRI'))
    print(task_func('2020-01-01', 5, 'WOM-2FRI', [200, 300, 400, 500, 600]))