import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = ['Date', 'Value']

def task_func(df, plot=False):
    if df.empty or not all(isinstance(lst, list) for lst in df['Value']):
        raise ValueError("DataFrame is empty or contains invalid 'Value' entries.")

    # Split lists in 'Value' column into separate columns
    value_df = pd.DataFrame(df['Value'].tolist(), index=df['Date'])
    value_df.reset_index(drop=True, inplace=True)

    # Calculate the Pearson correlation coefficient matrix
    corr_matrix = value_df.corr()

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap")
        plt.show()
        return corr_matrix, plt.gca()
    
    return corr_matrix