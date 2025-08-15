import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty DataFrame.")
    
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        raise ValueError("DataFrame must contain numeric columns.")
    
    axes = []
    for column in numeric_cols:
        ax = numeric_cols[column].hist(bins=20, edgecolor='k')
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        axes.append(ax)
    
    return axes