import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values")
    if not np.issubdtype(df.dtypes, np.number):
        raise TypeError("DataFrame contains non-numeric data types")
    
    scaler = MinMaxScaler()
    normalized_cumsums = pd.DataFrame()
    
    for column in df.columns:
        numeric_data = df[column].dropna()
        normalized_data = scaler.fit_transform(numeric_data.values.reshape(-1, 1)).flatten()
        normalized_cumsums[column] = np.cumsum(normalized_data)
        
    return normalized_cumsums