import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame must not be empty.")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=True, ax=axes[0])
    else:
        sns.histplot(df[col], ax=axes[0])

    sns.boxplot(x=df[col], ax=axes[1])

    axes[0].set_title(f'Histogram of {col}')
    axes[1].set_title(f'Box plot of {col}')

    plt.tight_layout()
    return fig