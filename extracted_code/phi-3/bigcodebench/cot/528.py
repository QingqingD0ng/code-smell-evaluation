import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("The file must be a CSV.")
    
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = list(reader)
    
    duplicates = Counter(tuple(row) for row in rows if rows.count(row) > 1)
    
    df_duplicates = pd.DataFrame(duplicates.items(), columns=header)
    df_duplicates = df_duplicates.drop_duplicates().reset_index(drop=True)
    
    counts = df_duplicates.set_index(header).iloc[:, 0].value_counts()
    
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    plt.tight_layout()
    return dict(duplicates), ax

# Usage example:
# duplicates, ax = task_func("sample_data.csv")
# print(duplicates)
# plt.show()