import pandas as pd
import numpy as np

def task_func(data_list=None):
    if data_list is None:
        return pd.DataFrame()
    data_list = np.array(data_list)
    mean_values = [np.nanmean([item[i] for item in data_list if isinstance(item[i], (int, float))]) for i in range(data_list.shape[1])]
    return pd.DataFrame({'Mean Value': mean_values}, index=[f'Position {i}' for i in range(len(mean_values))])