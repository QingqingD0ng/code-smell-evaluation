import json
import pandas as pd
import numpy as np

def calculate_stats(values):
    numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    mean_val = np.mean(numeric_values) if numeric_values else np.nan
    median_val = np.median(numeric_values) if numeric_values else np.nan
    return mean_val, median_val

def task_func(input_file="data.json"):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    stats = defaultdict(list)
    for record in data:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                stats[key].append(value)
    
    df = pd.DataFrame(index=stats.keys(), columns=['mean','median'])
    for key, values in stats.items():
        df.loc[key] = calculate_stats(values)
    
    return df