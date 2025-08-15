import json
import pandas as pd
import numpy as np
from collections import defaultdict

def task_func(input_file="data.json"):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    numeric_data = defaultdict(list)

    for record in data:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                numeric_data[key].append(value)
    
    results = {}
    for key, values in numeric_data.items():
        values_array = np.array(values)
        results[key] = {
           'mean': np.mean(values_array),
           'median': np.median(values_array)
        }
    
    df = pd.DataFrame(results).transpose()
    return df

# Example call
# df = task_func('data_1.json')