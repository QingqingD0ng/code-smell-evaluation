import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"File not found: {file_location}")
    
    try:
        data = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError(f"Sheet '{sheet_name}' does not exist in the file.")
    
    means = data.mean()
    std_devs = data.std()
    
    stats = {
       'mean': means.to_dict(),
       'std_dev': std_devs.to_dict()
    }
    
    fig, ax = plt.subplots()
    index = np.arange(len(means))
    bar_width = 0.35
    
    bars1 = ax.bar(index, means, bar_width, label='Mean')
    bars2 = ax.bar(index + bar_width, std_devs, bar_width, label='Std Dev')
    
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')
    ax.set_title('Mean and Standard Deviation')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(means.index)
    ax.legend()
    
    plt.tight_layout()
    return stats, fig