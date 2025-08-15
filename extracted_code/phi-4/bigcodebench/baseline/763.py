import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = defaultdict(lambda: {'mean': None,'median': None})
    
    for key in data[0]:
        values = [item[key] for item in data]
        results[key]['mean'] = np.mean(values)
        results[key]['median'] = np.median(values)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Key', 'Mean', 'Median'])
        for key, stats in results.items():
            writer.writerow([key, stats['mean'], stats['median']])

    return results