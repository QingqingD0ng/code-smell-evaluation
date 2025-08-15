import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if not data:
        return {'sum': 0,'mean': np.nan,'min': np.nan,'max': np.nan}, None

    headers = ['Age', 'Salary', 'Experience']
    
    if column not in headers:
        raise KeyError(f"Invalid column name '{column}'. Valid options are: {headers}")

    df = pd.DataFrame(data, columns=headers)
    
    column_data = df[column]
    stats = {
       'sum': column_data.sum(),
       'mean': column_data.mean(),
       'min': column_data.min(),
       'max': column_data.max()
    }
    
    if column == 'Age':
        labels = df['Age'].astype(str)
        values = df[column]
    else:
        labels = df.index.astype(str)
        values = df[column]
    
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    
    return stats, ax