import pandas as pd
import numpy as np

def task_func(data_list=None):
    if data_list is None:
        return pd.DataFrame()
    
    mean_values = []
    for i in range(len(data_list[0])):
        # Extract numerical values for the current position
        position_values = [item[i] for item in data_list if isinstance(item[i], (int, float))]
        if position_values:
            mean_values.append(np.nanmean(position_values))
        else:
            mean_values.append(np.nan)
    
    # Use DataFrame constructor with more pythonic approach
    return pd.DataFrame({'Mean Value': mean_values}, index=[f'Position {i}' for i in range(len(mean_values))])