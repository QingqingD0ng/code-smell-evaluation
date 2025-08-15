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
    cum_sum_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    for column in cum_sum_df.columns:
        cum_sum_df[column] = cum_sum_df[column].cumsum()

    return cum_sum_df