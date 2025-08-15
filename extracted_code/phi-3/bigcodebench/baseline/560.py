from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    df = pd.DataFrame([x.split('-') for x in data.split(',')], columns=['Year', 'Month', 'Value'])
    df['Month'] = pd.to_datetime(df['Month'] + '-' + df['Year'], format='%m-%Y').dt.strftime('%B')
    df['Value'] = df['Value'].astype(int)
    df = df.pivot(index='Month', columns='Year', values='Value')
    df.plot(kind='bar')
    plt.title(f'Monthly Data for {df.columns[0]}')
    plt.xlabel('Month')
    plt.ylabel('Value')
    return plt.gca()