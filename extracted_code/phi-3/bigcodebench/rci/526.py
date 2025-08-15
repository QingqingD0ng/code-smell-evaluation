import json
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_mean_median(values):
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    return np.mean(numeric_values), np.median(numeric_values)

def task_func(input_file="data.json"):
    df_stats = defaultdict(dict)
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading the file: {e}")
        return pd.DataFrame()

    for entry in data:
        for key, value in entry.items():
            if key not in df_stats:
                df_stats[key] = {'mean': np.nan,'median': np.nan}
            if isinstance(value, (int, float)):
                df_stats[key]['mean'] = calculate_mean_median([df_stats[key]['mean'], value])[0]
                df_stats[key]['median'] = calculate_mean_median([df_stats[key]['median'], value])[1]
            elif isinstance(value, (list, np.ndarray)):
                numeric_values = [v for v in value if isinstance(v, (int, float))]
                df_stats[key]['mean'], df_stats[key]['median'] = calculate_mean_median(numeric_values)

    df = pd.DataFrame.from_dict(df_stats, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'key'}, inplace=True)
    df.sort_values('key', inplace=True)
    return df

# Example usage:
# df = task_func('data_1.json')