from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    df = pd.DataFrame([x.split('-') for x in data.split(',')], columns=['year','month', 'value'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['value'] = df['value'].astype(int)

    df['date'] = pd.to_datetime(df[['year','month']].assign(DAY=1))
    df.set_index('date', inplace=True)

    ax = df['value'].plot(kind='bar', xticks=df.index, title=f'Monthly Data for {df["year"].iloc[0]}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    ax.set_xticklabels(df.index.strftime('%b'), rotation=0)

    return ax