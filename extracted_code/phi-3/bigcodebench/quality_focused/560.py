import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime


def task_func(data):
    df = pd.DataFrame([i.split('-') for i in data.split(',')], columns=['date', 'value']).apply(pd.to_datetime, format='%Y-%m')
    df['value'] = df['value'].astype(int)
    df.set_index('date', inplace=True)
    df.resample('M').sum().plot(kind='bar')
    plt.title(f'Monthly Data for {df.index.year.iloc[0]}')
    plt.xlabel('Month')
    plt.ylabel('Value')
    return plt.gca()