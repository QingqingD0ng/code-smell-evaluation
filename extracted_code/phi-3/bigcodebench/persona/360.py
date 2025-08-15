import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.isfile(file_location):
        raise FileNotFoundError('Excel file not found at specified path.')

    try:
        data = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError('Specified sheet does not exist in the workbook.')

    stats = data.agg(['mean','std'])
    stats_dict = stats.to_dict()

    fig, ax = plt.subplots()
    ax.bar(stats_dict.keys(), stats_dict['mean'], yerr=stats_dict['std'])
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')

    return stats_dict, fig