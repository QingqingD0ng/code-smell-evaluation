import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("The file must have a.csv extension")

    try:
        with open(file_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = list(reader)
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

    duplicates = []
    seen = set()
    for row in data:
        row_tuple = tuple(row)
        if row_tuple in seen:
            duplicates.append(row)
        else:
            seen.add(row_tuple)

    duplicates_df = pd.DataFrame(duplicates, columns=header)
    duplicates_count = duplicates_df.groupby(list(header)).size().reset_index(name='Count')

    counter = Counter(tuple(x) for x in duplicates_count.values.tolist())
    duplicates_dict = dict(counter)

    ax = duplicates_count.plot(kind='bar', x=header, y='Count', legend=False)
    plt.show()

    return duplicates_dict, ax