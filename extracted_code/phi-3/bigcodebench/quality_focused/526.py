import json

import pandas as pd

import numpy as np


def task_func(input_file="data.json"):
    # Load JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store statistics
    stats = defaultdict(lambda: {'mean': [],'median': []})

    # Process each record
    for record in data:
        for key, value in record.items():
            if isinstance(value, (int, float)):  # Check if value is numeric
                stats[key]['mean'].append(value)
                stats[key]['median'].append(value)

    # Calculate mean and median, ignoring non-numeric or missing values
    for key in stats:
        numeric_values = [val for val in stats[key]['mean'] if isinstance(val, (int, float))]
        if numeric_values:
            stats[key]['mean'] = np.mean(numeric_values)
            stats[key]['median'] = np.median(numeric_values)
        else:
            stats[key]['mean'] = np.nan
            stats[key]['median'] = np.nan

    # Convert dictionary to DataFrame
    df = pd.DataFrame(stats).transpose()

    # Rename columns
    df.columns = ['mean','median']

    # Sort DataFrame by index (keys)
    df.sort_index(inplace=True)

    return df