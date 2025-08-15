import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(df, bins=20):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty DataFrame.")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("DataFrame must contain numeric columns.")
    
    axes = []
    for column in numeric_df.columns:
        try:
            fig, ax = plt.subplots()
            numeric_df[column].dropna().hist(ax=ax, bins=bins)
            ax.set_title(column)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            axes.append(ax)
        except Exception as e:
            print(f"Error plotting column {column}: {e}")
    
    return axes