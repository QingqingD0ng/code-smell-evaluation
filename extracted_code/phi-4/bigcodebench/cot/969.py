import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isna().any().any():
        raise ValueError("DataFrame contains NaN values")
    
    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise TypeError("DataFrame contains non-numeric data types")
    
    result_df = df.apply(lambda x: x.cumsum())
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(result_df)
    
    return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)