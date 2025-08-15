import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = ['Date', 'Value']

def task_func(df, plot=False):
    if df.empty or not all(isinstance(i, list) and i for i in df['Value']):
        raise ValueError("DataFrame is empty or contains invalid 'Value' entries.")
    
    df_expanded = pd.DataFrame(df['Value'].tolist(), index=df['Date']).reset_index(drop=True)
    corr_df = df_expanded.corr()
    
    if plot:
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Correlation Heatmap')
        plt.show()
        return corr_df, ax
    
    return corr_df