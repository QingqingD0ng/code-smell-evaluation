import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        raise ValueError("Input df must be a non-empty DataFrame containing the specified column.")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df[col], kde=True, ax=axes[0])
    sns.boxplot(x=df[col], ax=axes[1])
    plt.tight_layout()
    return fig