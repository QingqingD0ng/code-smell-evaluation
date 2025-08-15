import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name, capsize=5, bar_color='skyblue'):
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"File not found at {file_location}")

    try:
        data = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError(f"Sheet '{sheet_name}' does not exist in the workbook.")

    # Filter numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate statistics
    stats = numeric_data.agg(['mean','std']).to_dict('index')

    # Create bar chart
    fig, ax = plt.subplots()
    means = [stats[col]['mean'] for col in numeric_data.columns]
    stds = [stats[col]['std'] for col in numeric_data.columns]

    ax.bar(numeric_data.columns, means, yerr=stds, capsize=capsize, color=bar_color)
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')

    return stats, fig