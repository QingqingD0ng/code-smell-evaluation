from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def task_func(data):
    entries = data.split(',')
    months = []
    values = []

    for entry in entries:
        date_str, value_str = entry.split('-')
        year, month = date_str.split('-')
        value = int(value_str)
        months.append(int(month))
        values.append(value)

    df = pd.DataFrame({'Month': months, 'Value': values})
    ax = df.plot(kind='bar', x='Month', y='Value', legend=False)
    ax.set_title(f'Monthly Data for {year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    
    return ax

# Example usage
data = '2022-01-100,2022-02-200,2022-03-150,2022-04-300,2022-05-250,2022-06-350,2022-07-400,2022-08-450,2022-09-500,2022-10-550,2022-11-600,2022-12-650'
ax = task_func(data)
plt.show()