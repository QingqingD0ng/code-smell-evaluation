import os

import pandas as pd

import numpy as np


def get_file_path(excel_file_path, file_name):
    return os.path.join(excel_file_path, file_name)


def read_excel_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Excel file {file_path} does not exist.")
    return pd.read_excel(file_path)


def calculate_statistics(data, column_name):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel file.")
    values = data[column_name].values
    return {
       'mean': np.mean(values),
       'median': np.median(values),
       'std_dev': np.std(values)
    }


def task_func(excel_file_path, file_name, column_name):
    file_path = get_file_path(excel_file_path, file_name)
    data = read_excel_file(file_path)
    stats = calculate_statistics(data, column_name)
    os.remove(file_path)
    return stats