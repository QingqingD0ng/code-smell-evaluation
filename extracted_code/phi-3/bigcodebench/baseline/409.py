import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    full_path = os.path.join(excel_file_path, file_name)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Excel file {full_path} does not exist.")

    try:
        data = pd.read_excel(full_path)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the Excel file.")

        values = data[column_name].values
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_dev_val = np.std(values)

        return {
           'mean': mean_val,
           'median': median_val,
           'std_dev': std_dev_val
        }
    finally:
        os.remove(full_path)