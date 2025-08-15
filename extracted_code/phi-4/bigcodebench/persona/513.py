import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(column, data):
    if not data:
        raise ValueError("Data list is empty.")
    
    allowed_columns = ['Steps', 'Calories Burned', 'Distance Walked']
    if column not in allowed_columns:
        raise KeyError(f"Invalid column '{column}'. Allowed columns are: {allowed_columns}")
    
    for entry in data:
        if len(entry)!= 4:
            raise ValueError("Each entry must contain exactly 4 elements: Date, Steps, Calories Burned, Distance Walked.")
        if any(isinstance(entry[i], int) and entry[i] < 0 for i in range(1, 4)):
            raise ValueError("Numeric values for Steps, Calories Burned, and Distance Walked must be non-negative.")
    
    df = pd.DataFrame(data, columns=['Date', 'Steps', 'Calories Burned', 'Distance Walked'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    column_data = df[column]
    stats = {
       'sum': column_data.sum(),
       'mean': column_data.mean(),
       'min': column_data.min(),
       'max': column_data.max()
    }
    
    fig, ax = plt.subplots()
    ax.plot(df['Date'], column_data)
    ax.set_title(f'Line Chart of {column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    
    return stats, ax