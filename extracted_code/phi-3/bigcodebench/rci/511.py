import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLUMN_INDICES = {'Age': 0, 'Salary': 1, 'Experience': 2}

def get_statistics(column_data):
    if not column_data:
        stats = {'sum': 0,'mean': np.nan,'min': np.nan,'max': np.nan}
    else:
        stats = {
          'sum': column_data.sum(),
          'mean': column_data.mean(),
          'min': column_data.min(),
          'max': column_data.max()
        }
    return stats

def plot_pie_chart(column_data):
    ax = column_data.value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%')
    return ax

def task_func(column_name, data):
    try:
        column_index = COLUMN_INDICES[column_name]
    except KeyError:
        raise ValueError(f"Invalid column name. Valid values are: {list(COLUMN_INDICES.keys())}")
    
    if not data:
        return {}, None
    
    column_data = pd.DataFrame(data)[:, column_index]
    stats = get_statistics(column_data)
    ax = plot_pie_chart(column_data)
    return stats, ax