import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        raise ValueError("Input must be a non-empty DataFrame with the specified column.")
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    sns.histplot(df[col], kde=True, ax=axes[0])
    sns.boxplot(x=df[col], ax=axes[1])
    
    plt.tight_layout()
    return fig

# Example usage:
# df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
# fig = task_func(df, 'value')
# plt.close()

# df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']})
# fig = task_func(df, 'category')
# plt.close()