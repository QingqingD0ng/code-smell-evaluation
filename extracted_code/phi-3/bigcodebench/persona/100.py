import matplotlib.pyplot as plt

import pandas as pd

import random

from datetime import datetime, timedelta


def task_func(seed=42):
    random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=30)
    values = [random.random() for _ in range(30)]
    data = pd.DataFrame({'Date': dates, 'Value': values})
    plt.style.use('seaborn-darkgrid')
    ax = data.plot(kind='line', x='Date', y='Value', fontname='Arial')
    ax.set_title('Random Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    return ax