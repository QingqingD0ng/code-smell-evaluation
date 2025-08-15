import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"The file {file_location} does not exist.")
    
    try:
        data = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError(f"The sheet {sheet_name} does not exist in the workbook.")
    
    mean_std = data.agg(['mean','std']).transpose()
    mean_std['mean_std'] = mean_std['mean'] + mean_std['std']
    
    fig, ax = plt.subplots()
    mean_std['mean_std'].plot(kind='bar', ax=ax)
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')
    
    return mean_std.to_dict(), fig