import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col, figsize=(12, 6), hist_kde=True, plot_titles=None):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        raise ValueError("Ensure df is a non-empty DataFrame containing the specified column.")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=hist_kde, ax=axes[0])
    else:
        sns.histplot(df[col], ax=axes[0])

    sns.boxplot(x=df[col], ax=axes[1])

    axes[0].set_title(plot_titles[0] if plot_titles else f'Histogram of {col}')
    axes[1].set_title(plot_titles[1] if plot_titles else f'Box plot of {col}')

    plt.tight_layout()
    return fig