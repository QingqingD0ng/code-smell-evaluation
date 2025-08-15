import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values")
    if not np.issubdtype(df.dtypes, np.number).all():
        raise TypeError("DataFrame contains non-numeric data types")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    normalized_cumsum = pd.DataFrame(scaled_data).cumsum(axis=0)
    normalized_cumsum.index = df.index
    normalized_cumsum.columns = df.columns
    return normalized_cumsum