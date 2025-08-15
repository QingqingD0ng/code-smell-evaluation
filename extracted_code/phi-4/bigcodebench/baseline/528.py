import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("File must be a CSV")

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    
    duplicates = Counter(tuple(row) for row in rows)
    duplicates = {k: v for k, v in duplicates.items() if v > 1}

    df_duplicates = pd.DataFrame(list(duplicates.items()), columns=['Row', 'Count'])
    df_duplicates['Row'] = df_duplicates['Row'].apply(lambda x: ', '.join(x))

    fig, ax = plt.subplots()
    ax.bar(df_duplicates['Row'], df_duplicates['Count'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return duplicates, ax