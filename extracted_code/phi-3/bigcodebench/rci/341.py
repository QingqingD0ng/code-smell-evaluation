import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col, col_type):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input df must not be empty.")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if col_type not in ['numerical', 'categorical']:
        raise ValueError("Column type must be 'numerical' or 'categorical'.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if col_type == 'numerical':
        sns.histplot(df[col], kde=True, ax=axes[0])
        sns.boxplot(x=df[col], ax=axes[1])
    else:
        sns.boxplot(x=df[col], ax=axes[0])
        sns.histplot(df[col], kde=True, ax=axes[1])

    plt.tight_layout()
    return fig