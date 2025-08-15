import json
import pandas as pd
import numpy as np
from collections import defaultdict

def task_func(input_file="data.json"):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    stats = defaultdict(list)
    
    if not isinstance(data, list):
        return pd.DataFrame()

    keys = set()
    for record in data:
        for key, value in record.items():
            keys.add(key)
            if isinstance(value, (int, float)):
                stats[key].append(value)
    
    result = {}
    for key in keys:
        values = np.array(stats[key])
        if values.size > 0:
            mean_val = np.mean(values)
            median_val = np.median(values)
        else:
            mean_val = np.nan
            median_val = np.nan
        result[key] = {'mean': mean_val,'median': median_val}

    df = pd.DataFrame(result).T
    return df.reindex(sorted(df.index))