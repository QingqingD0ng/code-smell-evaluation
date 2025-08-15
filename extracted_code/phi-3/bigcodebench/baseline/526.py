import json
import pandas as pd
import numpy as np
from collections import defaultdict

def task_func(input_file="data.json"):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    stats = defaultdict(lambda: {'mean': np.nan,'median': np.nan})
    
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                if np.isnan(stats[key]['mean']):
                    stats[key]['mean'] = value
                else:
                    stats[key]['mean'] = np.mean([stats[key]['mean'], value])
                values = [value]
            elif isinstance(value, (list, np.ndarray)) and all(isinstance(item, (int, float)) for item in value):
                values.extend(item for item in value if isinstance(item, (int, float)))
            elif isinstance(value, (list, np.ndarray)):
                values.extend(item for item in value if isinstance(item, (int, float)))
    
    for key, value in stats.items():
        values = [v for v in values if not np.isnan(v)]
        if values:
            stats[key]['median'] = np.median(values)
    
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'key'}, inplace=True)
    return df.sort_values('key')

# Example usage:
# df = task_func('data_1.json')