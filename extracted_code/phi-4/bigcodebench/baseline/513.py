import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(column, data):
    # Validate column
    allowed_columns = ['Date', 'Steps', 'Calories Burned', 'Distance Walked']
    if column not in allowed_columns:
        raise KeyError(f"Invalid column: {column}. Allowed columns are: {allowed_columns}")

    # Validate data
    if not data:
        raise ValueError("Data list must not be empty.")
    
    for entry in data:
        if len(entry)!= 4:
            raise ValueError("Each entry must contain a date followed by three numeric values.")
        if not isinstance(entry[0], datetime):
            raise ValueError("The first element of each entry must be a datetime object.")
        if any(x < 0 for x in entry[1:]):
            raise ValueError("Numeric values for steps, calories burned, and distance walked must be non-negative.")

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Date', 'Steps', 'Calories Burned', 'Distance Walked'])

    # Calculate statistics
    col_index = allowed_columns.index(column)
    column_data = df.iloc[:, col_index]
    stats = {
       'sum': column_data.sum(),
       'mean': column_data.mean(),
       'min': column_data.min(),
       'max': column_data.max()
    }

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df['Date'], column_data, marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    ax.set_title(f'Line Chart of {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return stats, ax