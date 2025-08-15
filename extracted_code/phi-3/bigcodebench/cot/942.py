import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.offsets import WeekOfMonth

def generate_dates(start_date, periods, freq):
    year = int(start_date[:4])
    month = int(start_date[5:7])
    week_of_month = 2 if 'WOM-2FRI' in freq else 1
    dates = pd.date_range(f'{year}-{month}-01', periods=periods, freq=freq)
    dates = dates + WeekOfMonth(week=week_of_month)
    return dates

def generate_sales_data(dates, categories):
    sales_data = []
    for date in dates:
        for category in categories:
            sales = np.random.randint(100, 500)
            sales_data.append([date, category, sales])
    return pd.DataFrame(sales_data, columns=['Date', 'Category', 'Sales'])

def plot_sales_report(df, ax):
    for category in df['Category'].unique():
        category_data = df[df['Category'] == category]
        ax.plot(category_data['Date'], category_data['Sales'], label=category)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Report')

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', categories=['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']):
    dates = generate_dates(start_date, periods, freq)
    df = generate_sales_data(dates, categories)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    fig, ax = plt.subplots()
    plot_sales_report(df,