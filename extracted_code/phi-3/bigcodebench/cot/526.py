import json
import pandas as pd
import numpy as np
from collections import defaultdict

def task_func(input_file="data.json"):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    stats = defaultdict(list)
    
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                stats[key].append(value)
    
    stats_df = pd.DataFrame({key: pd.Series(values) for key, values in stats.items()}).transpose()
    stats_df['mean'] = stats_df.mean(axis=1)
    stats_df['median'] = stats_df.median(axis=1)
    
    return stats_df