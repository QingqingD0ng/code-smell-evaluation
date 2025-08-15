import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_histograms(df, bins='auto', return_figures=False):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty DataFrame.")
    if not df.select_dtypes(include=np.number).shape[1]:
        raise ValueError("DataFrame must contain numeric columns.")
    numeric_cols = df.select_dtypes(include=np.number)
    if np.isinf(numeric_cols).any().any() or np.isnan(numeric_cols).any().any():
        raise TypeError("DataFrame contains non-numeric data types.")
        
    axes = []
    figs = []
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(numeric_cols[col].dropna(), bins=bins, color='blue', alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        axes.append(ax)
        if return_figures:
            figs.append((fig, ax))
    return figs if return_figures else axes

# Example usage:
# df = pd.DataFrame({'A': np.random.normal(0, 1, 100), 'B': np.random.exponential(1, 100)})
# figs = generate_histograms(df, bins=20, return_figures=True)
# for fig, _ in figs:
#     plt.show()