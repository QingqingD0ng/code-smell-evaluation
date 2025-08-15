import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("The file must have a.csv extension")

    with open(file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)

    duplicates = Counter(tuple(row) for row in data if data.count(row) > 1)
    duplicates_df = pd.DataFrame.from_records(duplicates.items(), columns=['Duplicate Rows', 'Count'])

    ax = duplicates_df.plot(kind='bar', x='Duplicate Rows', y='Count', legend=False)
    plt.show()

    return dict(duplicates), ax