import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values.")
    if not df.applymap(np.can_cast).all().all():
        raise TypeError("DataFrame contains non-numeric data types.")
    
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index)
    df_cumulative_sum = df_normalized.cumsum()
    
    return df_cumulative_sum