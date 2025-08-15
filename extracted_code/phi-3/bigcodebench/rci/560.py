from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    try:
        df = pd.DataFrame([x.split('-') for x in data.split(',')], columns=['Year', 'Month', 'Value'])
        df['Month'] = pd.to_datetime(df['Month'] + '-' + df['Year'], format='%b-%Y').dt.strftime('%B')
        df['Value'] = df['Value'].astype(int)
        df = df.pivot(index='Month', columns='Year', values='Value')
        ax = df.plot(kind='bar')
        ax.set_title(f'Monthly Data for {df.columns[0]}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        return ax
    except Exception as e:
        print(f'An error occurred: {e}')
        return None