from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    # Split the data string into a list of entries
    entries = data.split(',')

    # Parse the entries into a list of dictionaries
    records = []
    for entry in entries:
        date_str, value_str = entry.split('-')
        records.append({
            'date': datetime.strptime(date_str, '%Y-%m'),
            'value': int(value_str)
        })

    # Create a DataFrame from the records
    df = pd.DataFrame(records)

    # Set the date as the index
    df.set_index('date', inplace=True)

    # Resample the data by month and take the sum of values
    monthly_data = df.resample('M').sum()

    # Plot the data
    ax = monthly_data.plot(kind='bar', legend=False)

    # Set the title and labels
    ax.set_title(f'Monthly Data for {monthly_data.index[0].year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')

    return ax

# Example usage
data = '2022-01-100,2022-02-200,2022-03-150,2022-04-300,2022-05-250,2022-06-350,2022-07-400,2022-08-450,2022-09-500,2022-10-550,2022-11-600,2022-12-650'
ax = task_func(data)
plt.show()