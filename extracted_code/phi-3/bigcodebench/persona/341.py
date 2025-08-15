import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        raise ValueError("Input df must be a non-empty DataFrame containing the specified column.")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=True, ax=axes[0])
    else:
        sns.countplot(x=col, data=df, ax=axes[0])
    
    sns.boxplot(x=col, data=df, ax=axes[1])
    
    plt.tight_layout()
    return fig