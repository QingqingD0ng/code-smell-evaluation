import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df, plot=False):
    if df.empty or df['Value'].apply(lambda x: isinstance(x, list)).any() == False:
        raise ValueError("DataFrame is empty or 'Value' column contains invalid data.")

    df_split = df['Value'].apply(pd.Series)
    df_combined = pd.concat([df['Date'], df_split], axis=1)

    corr_matrix = df_split.corr()

    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Correlation Heatmap')
        plt.show()
        return corr_matrix, plt.gca()

    return corr_matrix

# Example usage
if __name__ == "__main__":
    df = pd.DataFrame([['2021-01-01', [8, 10, 12]], ['2021-01-02', [7, 9, 11]]], columns=['Date', 'Value'])
    corr_df = task_func(df)
    print(corr_df)