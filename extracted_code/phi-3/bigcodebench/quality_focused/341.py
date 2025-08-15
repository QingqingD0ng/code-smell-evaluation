import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        raise ValueError("Input must be a non-empty DataFrame containing the specified column.")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    if df[col].dtype.kind in 'iufc':  # Check if the column is numeric
        sns.histplot(df[col], kde=True, ax=axs[0])
        axs[0].set_title(f'Histogram and KDE of {col}')
    else:
        sns.boxplot(x=df[col], ax=axs[0])
        axs[0].set_title(f'Boxplot of {col}')
    
    sns.boxplot(x=df[col], ax=axs[1])
    axs[1].set_title(f'Boxplot of {col}')
    
    plt.tight_layout()
    return fig