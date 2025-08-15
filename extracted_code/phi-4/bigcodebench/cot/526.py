import json
import pandas as pd
import numpy as np

def task_func(input_file="data.json"):
    with open(input_file, 'r') as file:
        data = json.load(file)

    stats = {}

    for record in data:
        for key, value in record.items():
            if key not in stats:
                stats[key] = []
            if isinstance(value, (int, float)):
                stats[key].append(value)

    results = {}
    for key, values in stats.items():
        values = np.array(values)
        if values.size > 0:
            results[key] = {
                "mean": np.mean(values),
                "median": np.median(values)
            }
        else:
            results[key] = {
                "mean": np.nan,
                "median": np.nan
            }

    df = pd.DataFrame(results).T
    df.index.name = "variable"
    return df