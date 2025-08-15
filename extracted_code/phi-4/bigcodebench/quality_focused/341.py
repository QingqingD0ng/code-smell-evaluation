import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame must not be empty.")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(df[col], kde=True, ax=axs[0])
    axs[0].set_title('Histogram with KDE')

    sns.boxplot(x=df[col], ax=axs[1])
    axs[1].set_title('Box Plot')

    plt.tight_layout()
    return fig