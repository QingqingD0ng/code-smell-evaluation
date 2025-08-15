import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pandas.tseries.offsets import CustomBusinessDay


# Constants

START_DATE = '2016-01-01'

PERIODS = 13

FREQ = 'WOM-2FRI'

CATEGORIES = ['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']


def generate_dates(start_date, periods, freq):
    # Create a custom business day offset for the specific frequency
    bday_us = CustomBusinessDay(weekday=CustomBusinessDay.FRIDAY)
    date_range = pd.bdate_range(start_date, periods=periods, freq=freq).values

    return date_range


def task_func(start_date, periods, freq, categories):
    dates = generate_dates(start_date, periods, freq)
    categories = categories if isinstance(categories, list) else [categories]
    sales_data = pd.DataFrame(
        np.random.randint(100, 1000, size=(periods * len(categories), 3)),
        columns=['Sales', 'Category', 'Date']
    )
    sales_data['Date'] = pd.to_datetime(dates)
    sales_data['Category'] = np.repeat(categories, periods)
    sales_data = sales_data.sort_values(by=['Date', 'Category']).reset_index(drop=True)
    sales_data['Category'] = sales_data['Category'].astype('category')
    ax = sales_data.plot(kind='line', x='Date', y='Sales', figsize=(12, 6), xlabel='Date', ylabel='Sales', title='Sales Report')
    return sales_data, ax