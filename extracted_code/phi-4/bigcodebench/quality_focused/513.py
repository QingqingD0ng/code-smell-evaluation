import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if not data:
        raise ValueError("Data list is empty.")
    
    valid_columns = ['Date', 'Steps', 'Calories Burned', 'Distance Walked']
    
    if column not in valid_columns:
        raise KeyError(f"Invalid column '{column}'. Allowed columns are: {valid_columns}.")
    
    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=valid_columns)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Validate non-negative numeric values
    numeric_columns = ['Steps', 'Calories Burned', 'Distance Walked']
    for col in numeric_columns:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in '{col}' column.")
    
    # Calculate statistics
    stats = {
       'sum': df[column].sum(),
       'mean': df[column].mean(),
       'min': df[column].min(),
       'max': df[column].max()
    }
    
    # Plot line chart
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df[column])
    ax.set_title(f'Line Chart of {column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return stats, ax