import json
import pandas as pd
import numpy as np
from collections import defaultdict

def task_func(input_file="data.json"):
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        raise RuntimeError(f"Error reading file {input_file}")

    stats = defaultdict(list)

    for entry in data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                stats[key].append(value)

    results = {}
    for key, values in stats.items():
        values_array = np.array(values, dtype=float)
        mean_val = np.mean(values_array) if values_array.size > 0 else np.nan
        median_val = np.median(values_array) if values_array.size > 0 else np.nan
        results[key] = {'mean': mean_val,'median': median_val}

    df = pd.DataFrame.from_dict(results, orient='index')
    return df