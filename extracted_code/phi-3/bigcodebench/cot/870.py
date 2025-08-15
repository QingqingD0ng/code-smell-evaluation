import pandas as pd
import numpy as np


def task_func(data_list=None):
    if data_list is None:
        return pd.DataFrame()

    if not all(isinstance(t, tuple) for t in data_list):
        raise ValueError("All elements of data_list must be tuples.")

    numerical_values = np.array([val for t in data_list for val in t if isinstance(val, (int, float))])
    if numerical_values.size == 0:
        return pd.DataFrame()

    means = np.mean(numerical_values.reshape(-1, len(data_list[0])), axis=0)
    df = pd.DataFrame({'Mean Value': means}, index=[f'Position {i}' for i in range(len(data_list[0]))])
    return df