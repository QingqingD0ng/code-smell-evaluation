import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = ['Date', 'Value']

def task_func(df, plot=False):
    if df.empty:
        raise ValueError("DataFrame is empty")
    if not all(isinstance(val, list) for val in df['Value']):
        raise ValueError("Invalid 'Value' column")
    
    df_expanded = pd.DataFrame(df['Value'].tolist(), index=df.index)
    corr_matrix = df_expanded.corr()
    
    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
        return corr_matrix, plt.gcf()  # Return the Axes object from plt.gcf()
    else:
        return corr_matrix