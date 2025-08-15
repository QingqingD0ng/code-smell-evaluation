import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    field_stats = defaultdict(list)
    for item in data:
        for key, value in item.items():
            field_stats[key].append(value)
    
    field_mean_median = {}
    for key, values in field_stats.items():
        field_mean_median[key] = {'mean': np.mean(values),'median': np.median(values)}
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['field','mean','median'])
        writer.writeheader()
        for key, stats in field_mean_median.items():
            writer.writerow({'field': key,'mean': stats['mean'],'median': stats['median']})