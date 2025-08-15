import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_data(column, data):
    if column not in ['Date', 'Steps', 'Calories Burned', 'Distance Walked']:
        raise KeyError('Invalid column')
    if not data:
        raise ValueError('Data list is empty')
    df = pd.DataFrame(data, columns=['Date', column])
    df[column] = df[column].apply(lambda x: 0 if x < 0 else x)
    return df

def get_statistics(df, column):
    stats = {
       'sum': df[column].sum(),
       'mean': df[column].mean(),
       'min': df[column].min(),
       'max': df[column].max()
    }
    return stats

def plot_data(df, column):
    df['Date'] = pd.to_datetime(df['Date'])
    ax = df.plot(x='Date', y=column, title=f'Line Chart of {column}')
    return ax

def task_func(column, data):
    df = process_data(column, data)
    stats = get_statistics(df, column)
    ax = plot_data(df, column)
    return stats, ax