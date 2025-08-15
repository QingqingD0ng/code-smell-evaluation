import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(column, data):
    if not data:
        raise ValueError("Data list must not be empty.")
    
    allowed_columns = ['Date', 'Steps', 'Calories Burned', 'Distance Walked']
    if column not in allowed_columns:
        raise KeyError(f"Invalid column. Allowed columns are: {allowed_columns}")

    column_indices = {
        'Date': 0,
        'Steps': 1,
        'Calories Burned': 2,
        'Distance Walked': 3
    }

    numeric_columns = ['Steps', 'Calories Burned', 'Distance Walked']
    for row in data:
        for num_col in numeric_columns:
            if row[column_indices[num_col]] < 0:
                raise ValueError(f"Numeric values for {num_col} must be non-negative.")

    df = pd.DataFrame(data, columns=allowed_columns)
    df['Date'] = pd.to_datetime(df['Date'])

    if column!= 'Date':
        column_data = df[column].astype(float)
        stats = {
           'sum': column_data.sum(),
           'mean': column_data.mean(),
           'min': column_data.min(),
           'max': column_data.max()
        }
    else:
        stats = {
           'sum': None,
           'mean': None,
           'min': df['Date'].min(),
           'max': df['Date'].max()
        }

    fig, ax = plt.subplots()
    ax.plot(df['Date'], df[column].astype(float))
    ax.set_title(f'Line Chart of {column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)

    return stats, ax