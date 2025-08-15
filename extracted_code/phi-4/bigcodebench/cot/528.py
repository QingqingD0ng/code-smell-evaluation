import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("File must be a CSV with.csv extension.")
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    duplicates = Counter(tuple(row) for row in rows)
    duplicates = {k: v for k, v in duplicates.items() if v > 1}
    
    if not duplicates:
        raise ValueError("No duplicate rows found.")
    
    df = pd.DataFrame(list(duplicates.items()), columns=['Row', 'Count'])
    
    fig, ax = plt.subplots()
    ax.bar(range(len(duplicates)), df['Count'], tick_label=[str(row) for row in df['Row']])
    ax.set_xlabel('Duplicate Rows')
    ax.set_ylabel('Count')
    ax.set_title('Duplicate Rows in CSV')
    
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    return duplicates, ax