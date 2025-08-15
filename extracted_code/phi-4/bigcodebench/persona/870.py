import pandas as pd
import numpy as np

def task_func(data_list=[('a', 1, 2.1), ('b', 2, 3.2), ('c', 3, 4.3), ('d', 4, 5.4), ('e', 5, 6.5)]):
    if not data_list:
        return pd.DataFrame()

    # Transpose the list to get columns
    transposed_data = list(zip(*data_list))
    
    # Calculate means for each position, ignoring non-numeric values
    mean_values = []
    for position, values in enumerate(transposed_data):
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if numeric_values:
            mean = np.mean(numeric_values)
        else:
            mean = np.nan
        mean_values.append(mean)
    
    # Create DataFrame
    df = pd.DataFrame(mean_values, index=[f'Position {i}' for i in range(len(mean_values))], columns=['Mean Value'])
    return df