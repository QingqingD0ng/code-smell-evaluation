import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if not data:
        return {'sum': 0,'mean': np.nan,'min': np.nan,'max': np.nan}, None
    
    df = pd.DataFrame(data, columns=['Age', 'Salary', 'Experience'])
    
    if column not in df.columns:
        raise KeyError(f"Invalid column: {column}")
    
    col_data = df[column]
    col_sum = col_data.sum()
    col_mean = col_data.mean()
    col_min = col_data.min()
    col_max = col_data.max()
    
    stats = {
       'sum': col_sum,
       'mean': col_mean,
       'min': col_min,
       'max': col_max
    }
    
    plt.figure()
    if column == 'Age':
        plt.pie(col_data, labels=col_data, autopct='%1.1f%%', startangle=140)
    else:
        plt.pie(col_data, autopct='%1.1f%%', startangle=140)
    
    ax = plt.gca()
    plt.show()
    
    return stats, ax