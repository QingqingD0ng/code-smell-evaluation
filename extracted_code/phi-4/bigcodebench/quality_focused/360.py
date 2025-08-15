import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"File not found: {file_location}")
    
    try:
        df = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError(f"Sheet not found: {sheet_name}")
    
    stats = df.agg(['mean','std']).transpose()
    means = stats['mean']
    stds = stats['std']
    
    fig, ax = plt.subplots()
    ax.bar(means.index, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')
    
    return stats.to_dict('index'), fig