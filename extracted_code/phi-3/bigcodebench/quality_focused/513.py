import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def task_func(column, data):
    if column not in ['Date', 'Steps', 'Calories Burned', 'Distance Walked']:
        raise KeyError("Invalid column name.")
    
    if not data:
        raise ValueError("Data list is empty.")
    
    df = pd.DataFrame(data, columns=['Date', 'Steps', 'Calories Burned', 'Distance Walked'])
    if (df['Steps'] < 0).any() or (df['Calories Burned'] < 0).any() or (df['Distance Walked'] < 0).any():
        raise ValueError("Negative values found for steps, calories burned, or distance walked.")
    
    stats = {
       'sum': df[column].sum(),
       'mean': df[column].mean(),
       'min': df[column].min(),
       'max': df[column].max()
    }
    
    ax = df.plot(x='Date', y=column, kind='line', title=f'Line Chart of {column}')
    plt.show()
    
    return stats, ax