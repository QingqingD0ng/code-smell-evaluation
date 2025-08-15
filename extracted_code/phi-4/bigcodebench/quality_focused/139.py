import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty DataFrame.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if numeric_cols.empty:
        raise ValueError("No numeric columns found in the DataFrame.")
    
    axes = []
    for col in numeric_cols:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        axes.append(ax)
    
    return axes