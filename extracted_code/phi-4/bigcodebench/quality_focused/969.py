import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def task_func(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values.")
    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise TypeError("DataFrame contains non-numeric data types.")

    cumulative_sum = df.cumsum()
    scaler = MinMaxScaler()
    normalized_cumulative_sum = pd.DataFrame(scaler.fit_transform(cumulative_sum), 
                                             index=cumulative_sum.index, 
                                             columns=cumulative_sum.columns)
    return normalized_cumulative_sum