import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    stats = defaultdict(lambda: {'mean': [],'median': []})
    
    for record in data:
        for key, value in record.items():
            stats[key]['mean'].append(value)
    
    results = {}
    for key, values in stats.items():
        results[key] = {
           'mean': np.mean(values['mean']),
           'median': np.median(values['mean'])
        }
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key','mean','median'])
        for key, value in results.items():
            writer.writerow([key, value['mean'], value['median']])