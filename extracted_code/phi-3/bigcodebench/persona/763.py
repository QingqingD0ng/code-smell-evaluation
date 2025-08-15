import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)

    stats = defaultdict(dict)
    for key in data[0].keys():
        values = [entry[key] for entry in data]
        stats[key]['mean'] = np.mean(values)
        stats[key]['median'] = np.median(values)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['key','mean','median'])
        writer.writeheader()
        for key, value in stats.items():
            writer.writerow({'key': key,'mean': value['mean'],'median': value['median']})

    return stats