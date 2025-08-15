import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty DataFrame.")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        raise ValueError("DataFrame must contain numeric columns.")
    
    axes = []
    for col in numeric_cols:
        plt.figure()
        axes.append(plt.hist(df[col], bins='auto', color='blue', alpha=0.7))
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    return axes