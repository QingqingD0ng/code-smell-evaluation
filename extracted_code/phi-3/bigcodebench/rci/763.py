import numpy as np
import json
from collections import defaultdict
import csv

def calculate_statistics(values):
    return {
      'mean': np.mean(values),
      'median': np.median(values)
    }

def task_func(input_file, output_file):
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)
    
    stats = defaultdict(list)
    
    for record in data:
        for key, value in record.items():
            stats[key].append(value)
    
    results = {key: calculate_statistics(values) for key, values in stats.items()}
    
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['key','mean','median'])
        for key, value in results.items():
            writer.writerow([key, value['mean'], value['median']])