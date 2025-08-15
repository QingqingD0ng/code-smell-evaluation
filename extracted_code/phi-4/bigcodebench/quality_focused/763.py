import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = defaultdict(lambda: {'mean': None,'median': None})
    values_by_key = defaultdict(list)

    for entry in data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):  # Ensure the value is a number
                values_by_key[key].append(value)
    
    for key, values in values_by_key.items():
        np_values = np.array(values)
        results[key]['mean'] = np.mean(np_values)
        results[key]['median'] = np.median(np_values)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Field', 'Mean', 'Median'])
        for key, stats in results.items():
            writer.writerow([key, stats['mean'], stats['median']])
    
    return dict(results)