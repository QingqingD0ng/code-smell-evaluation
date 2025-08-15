import csv

from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt


def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("File must have a.csv extension")
    
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        data = [row for row in reader]
    
    duplicates_counter = Counter(tuple(row) for row in data if data.count(row) > 1)
    duplicates_df = pd.DataFrame.from_records(duplicates_counter.items(), columns=headers)
    
    fig, ax = plt.subplots()
    ax.bar(duplicates_df[headers[0]], duplicates_df[headers[1]])
    ax.set_xlabel(headers[0])
    ax.set_ylabel('Count')
    ax.set_title('Duplicate Rows Count')
    
    return dict(duplicates_counter), ax