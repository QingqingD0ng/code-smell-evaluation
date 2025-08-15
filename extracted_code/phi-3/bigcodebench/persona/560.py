import matplotlib.pyplot as plt

import pandas as pd

from datetime import datetime


def task_func(data):
    # Split the data string into a list of 'yyyy-mm-value' strings
    data_list = data.split(',')

    # Convert the list into a DataFrame with'month' and 'value' columns
    df = pd.DataFrame(data_list, columns=['month_value'])
    df[['month', 'value']] = df['month_value'].str.split('-', expand=True)
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m').dt.month
    df['value'] = df['value'].astype(int)

    # Group by month and sum the values
    df = df.groupby('month').sum().reset_index()

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(df['month'], df['value'], color='skyblue')
    ax.set_title(f'Monthly Data for {df.iloc[0]["month"].year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(range(1, 13))

    return ax


# Example usage:

data = '2022-01-100,2022-02-200,2022-03-150,2022-04-300,2022-05-250,2022-06-350,2022-07-400,2022-08-450,2022-09-500,2022-10-550,2022-11-600,2022-12-650'
ax = task_func(data)

# Show the plot

plt.show()