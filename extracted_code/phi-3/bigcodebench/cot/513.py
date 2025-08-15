import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(column, data):
    if column not in ['Date', 'Steps', 'Calories Burned', 'Distance Walked']:
        raise KeyError("Invalid column specified")

    if not data or not all(isinstance(item, list) and len(item) == 4 for item in data):
        raise ValueError("Data list must not be empty and must contain lists of length 4")

    for row in data:
        if not all(isinstance(value, (int, float)) and value >= 0 for value in row[1:]):
            raise ValueError("Numeric values must be non-negative")

    df = pd.DataFrame(data, columns=['Date', column, f'{column}_Burned', f'{column}_Distance'])

    stats = {
      'sum': df[column].sum(),
      'mean': df[column].mean(),
      'min': df[column].min(),
      'max': df[column].max()
    }

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    df.plot(x='Date', y=column, kind='line', ax=ax)
    plt.title(f'Line Chart of {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)

    return stats, ax