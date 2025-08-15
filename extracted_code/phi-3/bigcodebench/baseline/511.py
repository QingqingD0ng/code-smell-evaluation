import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if column not in ['Age', 'Salary', 'Experience']:
        raise KeyError("Invalid column. Valid values are 'Age', 'Salary', and 'Experience'.")
    df = pd.DataFrame(data, columns=[column, 'Salary', 'Experience'])
    stats = {
       'sum': df[column].sum(),
       'mean': df[column].mean(),
       'min': df[column].min(),
       'max': df[column].max()
    }
    if not data:
        stats['mean'] = np.nan
        stats['min'] = np.nan
        stats['max'] = np.nan
    ax = df[column].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%')
    return stats, ax