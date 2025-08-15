import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

COLUMNS = ['Date', 'Value']

def task_func(df, plot=False):
    if df.empty or not all(isinstance(val, list) for val in df['Value']):
        raise ValueError("DataFrame is empty or 'Value' column contains invalid data")

    # Split the 'Value' column into separate columns
    df_expanded = pd.DataFrame(df['Value'].tolist(), index=df.index).stack().reset_index()
    df_expanded.columns = ['Date', 'Index', 'Value']
    df_expanded['Date'] = df_expanded['Date'].astype(pd.Timestamp)
    
    # Calculate Pearson correlation coefficient
    corr_matrix = np.corrcoef(df_expanded['Value'].values.T)
    
    if plot:
        # Create a heatmap plot
        sns.set(style="white")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.show()
        return None, ax
    else:
        # Return the correlation coefficients as a DataFrame
        corr_df = pd.DataFrame(corr_matrix, columns=df_expanded['Value'].apply(lambda x: f"Col_{len(x)}"))
        corr_df.index = df_expanded['Value'].apply(lambda x: f"Col_{len(x)}")
        corr_df.columns = df_expanded['Value'].apply(lambda x: f"Col_{len(x)}")
        return corr_df, None