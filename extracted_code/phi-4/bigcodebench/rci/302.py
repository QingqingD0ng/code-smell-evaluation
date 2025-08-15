import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty or 'Date' not in df.columns or 'Value' not in df.columns:
        raise ValueError("DataFrame must be non-empty and contain 'Date' and 'Value' columns.")
    if not all(isinstance(lst, list) for lst in df['Value']):
        raise ValueError("All entries in 'Value' must be lists.")
    if not all(len(lst) == len(df['Value'][0]) for lst in df['Value']):
        raise ValueError("All lists in 'Value' must be of the same length.")

def split_value_column(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df['Value'].tolist(), index=df['Date']).reset_index(drop=True)

def calculate_correlation(value_df: pd.DataFrame) -> pd.DataFrame:
    return value_df.corr()

def plot_heatmap(corr_matrix: pd.DataFrame, figsize: tuple = (10, 8)) -> plt.Axes:
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    plt.show()
    return plt.gca()

def task_func(df: pd.DataFrame, plot: bool = False, figsize: tuple = (10, 8)) -> pd.DataFrame:
    validate_dataframe(df)
    value_df = split_value_column(df)
    corr_matrix = calculate_correlation(value_df)
    if plot:
        return corr_matrix, plot_heatmap(corr_matrix, figsize)
    return corr_matrix