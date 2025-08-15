import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isna().values.any():
        raise ValueError("DataFrame contains NaN values")
    
    if not np.issubdtype(df.dtypes, np.number).all():
        raise TypeError("DataFrame contains non-numeric data types")
    
    cumulative_sum = df.cumsum()
    scaler = MinMaxScaler()
    normalized_cumulative_sum = pd.DataFrame(scaler.fit_transform(cumulative_sum), columns=df.columns, index=df.index)
    
    return normalized_cumulative_sum