import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr


def task_func(df, plot=False):
    if df.empty:
        raise ValueError("DataFrame is empty")
    if not all(isinstance(val, list) for val in df['Value']):
        raise ValueError("Invalid 'Value' column content")
    
    df_exploded = df.explode('Value')
    df_exploded['Value'] = df_exploded['Value'].astype(float)
    df_exploded.index = pd.to_datetime(df_exploded['Date'])
    df_exploded = df_exploded.drop(columns=['Date'])
    
    corr_matrix = df_exploded.corr(method='pearson')
    
    if plot:
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()
    
    return corr_matrix