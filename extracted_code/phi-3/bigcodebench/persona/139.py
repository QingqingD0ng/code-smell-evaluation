import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def task_func(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty DataFrame.")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("DataFrame must contain numeric columns.")
    axes = []
    for col in numeric_cols:
        ax = df[col].hist()
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        axes.append(ax)
    return axes


# Example usage:

# df = pd.DataFrame({'A': np.random.normal(0, 1, 100), 'B': np.random.exponential(1, 100)})
# axes = task_func(df)
# for ax in axes:
#     plt.show()