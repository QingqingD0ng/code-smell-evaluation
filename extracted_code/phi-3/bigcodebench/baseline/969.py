import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values.")
    if not np.issubdtype(df.dtypes, np.number):
        raise TypeError("DataFrame contains non-numeric data types.")

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    df_cumulative_sum = df_normalized.cumsum()
    
    return df_cumulative_sum