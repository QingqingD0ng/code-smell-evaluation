import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    values_dict = defaultdict(list)
    
    for entry in data:
        for key, value in entry.items():
            values_dict[key].append(value)
    
    stats = {}
    for key, values in values_dict.items():
        mean = np.mean(values)
        median = np.median(values)
        stats[key] = {'mean': mean,'median': median}
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['key','mean','median']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for key, stat in stats.items():
            writer.writerow({'key': key,'mean': stat['mean'],'median': stat['median']})
    
    return stats