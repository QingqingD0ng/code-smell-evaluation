import pandas as pd
import numpy as np

def task_func(data_list=[('a', 1, 2.1), ('b', 2, 3.2), ('c', 3, 4.3), ('d', 4, 5.4), ('e', 5, 6.5)]):
    numeric_data = [[t[i] for t in data_list if isinstance(t[i], (int, float))] for i in range(len(data_list[0]))]
    mean_values = [np.nanmean(col) if len(col) > 0 else np.nan for col in numeric_data]
    df = pd.DataFrame({'Mean Value': mean_values}, index=[f'Position {i}' for i in range(len(mean_values))])
    return df