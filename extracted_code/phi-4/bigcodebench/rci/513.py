import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(column, data):
    ALLOWED_COLUMNS = ['Date', 'Steps', 'Calories Burned', 'Distance Walked']
    EXPECTED_ENTRY_LENGTH = 4

    if column not in ALLOWED_COLUMNS:
        raise KeyError(f"Invalid column: {column}. Allowed columns are: {ALLOWED_COLUMNS}")

    if not data:
        raise ValueError("Data list must not be empty.")
    
    for entry in data:
        if len(entry)!= EXPECTED_ENTRY_LENGTH:
            raise ValueError(f"Each entry must contain exactly {EXPECTED_ENTRY_LENGTH} elements.")
        if not isinstance(entry[0], datetime):
            raise ValueError("The first element of each entry must be a datetime object.")
        if any(x < 0 for x in entry[1:]):
            raise ValueError("Numeric values for steps, calories burned, and distance walked must be non-negative.")

    df = pd.DataFrame(data, columns=ALLOWED_COLUMNS)

    column_data = df[column]
    stats = {
       'sum': column_data.sum(),
       'mean': column_data.mean(),
       'min': column_data.min(),
       'max': column_data.max()
    }

    fig, ax = plt.subplots()
    ax.plot(df['Date'], column_data, marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    ax.set_title(f'Line Chart of {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return stats, ax