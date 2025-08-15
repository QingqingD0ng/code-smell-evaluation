from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    df = pd.read_csv(StringIO(data), names=['date', 'value'], parse_dates=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    year = df['year'].iloc[0]
    df = df[df['year'] == year]
    df.set_index('month', inplace=True)
    ax = df['value'].plot(kind='bar', figsize=(10, 6), title=f'Monthly Data for {year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.show()
    return ax