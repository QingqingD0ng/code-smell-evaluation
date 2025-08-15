import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def task_func(file_location, sheet_name):
    if not os.path.isfile(file_location):
        raise FileNotFoundError(f"File {file_location} not found.")

    try:
        df = pd.read_excel(file_location, sheet_name=sheet_name)
    except ValueError:
        raise ValueError(f"Sheet {sheet_name} not found in the workbook.")

    results = df.agg(['mean','std']).transpose().reset_index().rename(columns={'index': 'Column'})
    results['Values'] = results.apply(lambda row: f"{row['mean']:.2f}±{row['std']:.2f}", axis=1)

    fig, ax = plt.subplots()
    ax.bar(results['Column'], results['Values'])
    ax.set_title('Mean and Standard Deviation')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')

    return results, fig