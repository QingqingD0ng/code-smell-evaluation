import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if column not in ['Date', 'Steps', 'Calories Burned', 'Distance Walked']:
        raise KeyError('Invalid column')
    if not data:
        raise ValueError('Data list is empty')
    df = pd.DataFrame(data, columns=['Date', column])
    df[column] = df[column].apply(lambda x: x if pd.notnull(x) else 0)
    if (df[column] < 0).any():
        raise ValueError('Numeric values must be non-negative')
    stats = {
       'sum': df[column].sum(),
       'mean': df[column].mean(),
       'min': df[column].min(),
       'max': df[column].max()
    }
    df['Date'] = pd.to_datetime(df['Date'])
    ax = df.plot(x='Date', y=column, title=f'Line Chart of {column}')
    return stats, ax