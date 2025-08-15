import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = ['Date', 'Value']

def task_func(df, plot=False):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    if not all(isinstance(val, list) for val in df['Value']):
        raise ValueError("All values in the 'Value' column must be lists.")
    
    # Split lists in 'Value' column into separate columns
    value_df = pd.DataFrame(df['Value'].tolist(), index=df['Date'])
    
    # Calculate the Pearson correlation coefficient matrix
    corr_matrix = value_df.corr()
    
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
        return corr_matrix, plt.gca()
    
    return corr_matrix

# Example usage
df = pd.DataFrame([['2021-01-01', [8, 10, 12]], ['2021-01-02', [7, 9, 11]]], columns=COLUMNS)
corr_df = task_func(df)
print(corr_df)