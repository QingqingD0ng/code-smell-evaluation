import numpy as np
from collections import defaultdict
import json
import csv

def task_func(input_file, output_file):
    # Read JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Initialize a defaultdict to store lists of values for each key
    values_by_key = defaultdict(list)

    # Populate the defaultdict with values from each dictionary
    for entry in data:
        for key, value in entry.items():
            values_by_key[key].append(value)

    # Prepare the results dictionary
    results = {}
    for key, values in values_by_key.items():
        # Convert list to numpy array for mean and median calculation
        values_array = np.array(values)
        results[key] = {
           'mean': float(np.mean(values_array)),
           'median': float(np.median(values_array))
        }

    # Write results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Mean', 'Median'])
        for key, stats in results.items():
            writer.writerow([key, stats['mean'], stats['median']])

    return results