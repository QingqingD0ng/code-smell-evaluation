import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def task_func(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("File must be a CSV.")
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Identify duplicates
    row_counts = Counter(tuple(row) for row in rows)
    duplicates = {k: v for k, v in row_counts.items() if v > 1}
    
    # Convert duplicates to DataFrame
    df_duplicates = pd.DataFrame(list(duplicates.keys()), columns=rows[0])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    ax = df_duplicates.iloc[:, 0].value_counts().plot(kind='bar')
    ax.set_title('Duplicate Rows Count')
    ax.set_xlabel('Duplicate Row (First Column)')
    ax.set_ylabel('Count')
    
    return duplicates, ax

# Example usage
# duplicates, ax = task_func("sample_data.csv")