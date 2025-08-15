import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.isfile(file_location):
        raise FileNotFoundError(f"File not found: {file_location}")
    try:
        data = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError(f"Sheet '{sheet_name}' not found in workbook.")
    stats = data.agg(['mean','std']).transpose()
    fig, ax = plt.subplots()
    ax.bar(stats.index, stats['mean'], yerr=stats['std'])
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')
    return stats.to_dict(), fig