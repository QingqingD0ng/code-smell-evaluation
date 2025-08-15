import pandas as pd
import numpy as np

def task_func(data_list=[('a', 1, 2.1), ('b', 2, 3.2), ('c', 3, 4.3), ('d', 4, 5.4), ('e', 5, 6.5)]):
    if not data_list:
        return pd.DataFrame(columns=['Mean Value'])
    
    num_columns = max(len(item) for item in data_list)
    means = []
    
    for i in range(num_columns):
        values = [item[i] for item in data_list if isinstance(item[i], (int, float))]
        mean_val = np.mean(values) if values else np.nan
        means.append(mean_val)
    
    index = [f'Position {i}' for i in range(num_columns)]
    df = pd.DataFrame(means, index=index, columns=['Mean Value'])
    
    return df