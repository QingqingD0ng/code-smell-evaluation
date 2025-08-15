import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"File not found at {file_location}")

    try:
        data = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError as e:
        raise ValueError(f"Sheet '{sheet_name}' does not exist in the workbook.") from e

    stats = {}
    for column in data.columns:
        stats[column] = {
           'mean': data[column].mean(),
           'std': data[column].std()
        }

    means = [stats[col]['mean'] for col in data.columns]
    stds = [stats[col]['std'] for col in data.columns]

    fig, ax = plt.subplots()
    ax.bar(data.columns, means, yerr=stds, capsize=5, color='skyblue')
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')

    return stats, fig