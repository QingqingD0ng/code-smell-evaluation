import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")

    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1]!= df.shape[1]:
        raise TypeError("DataFrame contains non-numeric data types")

    cumsum_df = numeric_df.cumsum()
    scaler = MinMaxScaler()
    normalized_cumsum = pd.DataFrame(scaler.fit_transform(cumsum_df), columns=numeric_df.columns, index=df.index)

    return normalized_cumsum