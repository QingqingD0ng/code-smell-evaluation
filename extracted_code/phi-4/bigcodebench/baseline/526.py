import json
import pandas as pd
import numpy as np
from collections import defaultdict

def task_func(input_file="data.json"):
    with open(input_file, 'r') as file:
        data = json.load(file)

    stats = defaultdict(lambda: {'values': []})

    for entry in data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                stats[key]['values'].append(value)

    results = {}
    for key, value_dict in stats.items():
        values = np.array(value_dict['values'], dtype=np.float64)
        if len(values) > 0:
            mean_val = np.mean(values)
            median_val = np.median(values)
        else:
            mean_val = np.nan
            median_val = np.nan
        results[key] = {'mean': mean_val,'median': median_val}

    df = pd.DataFrame(results).T
    return df