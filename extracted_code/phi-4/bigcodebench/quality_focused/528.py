import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("The file must have a.csv extension")

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    duplicates = Counter(tuple(row) for row in rows)
    duplicates = {k: v for k, v in duplicates.items() if v > 1}

    df = pd.DataFrame(list(duplicates.items()), columns=['Row', 'Count'])
    
    ax = df.set_index('Row')['Count'].plot(kind='bar')
    plt.ylabel('Count')
    plt.title('Duplicate Rows Count')
    
    return duplicates, ax