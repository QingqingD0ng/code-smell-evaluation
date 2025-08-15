import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)

    stats = defaultdict(lambda: {'mean': None,'median': None})
    for key in data[0].keys():
        values = [d[key] for d in data]
        stats[key]['mean'] = np.mean(values)
        stats[key]['median'] = np.median(values)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key','mean','median'])
        for key, value in stats.items():
            writer.writerow([key, value['mean'], value['median']])