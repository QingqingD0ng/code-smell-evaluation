from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    data_list = data.split(',')
    df = pd.DataFrame([
        {'month': datetime.strptime(item.split('-')[1], '%m'), 'value': int(item.split('-')[2])}
        for item in data_list], columns=['month', 'value'])
    df['month'] = df['month'].dt.month_name()
    plt.figure(figsize=(10, 6))
    ax = df.plot.bar(x='month', y='value', rot=0)
    ax.set_title(f'Monthly Data for {df["month"].iloc[0]}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    plt.tight_layout()
    return ax