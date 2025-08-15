import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

COLUMNS = ['Date', 'Value']

def task_func(df, plot=False):
    if df.empty or not all(isinstance(x, list) and all(isinstance(y, (int, float)) for y in x) for x in df['Value']):
        raise ValueError("DataFrame input is invalid or empty.")
        
    df_exploded = df.explode('Value')
    df_exploded['Value'] = pd.to_numeric(df_exploded['Value'])
    df_pivot = df_exploded.pivot_table(index='Date', columns='Value', aggfunc='size', fill_value=0)
    
    corr_matrix = df_pivot.corr(method='pearson')
    
    if plot:
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.show()
        
    return corr_matrix