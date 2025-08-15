import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    full_file_path = os.path.join(excel_file_path, file_name)
    if not os.path.isfile(full_file_path):
        raise FileNotFoundError(f"File not found: {full_file_path}")
    df = pd.read_excel(full_file_path)
    if column_name not in df.columns:
        raise ValueError(f"Column not found: {column_name}")
    data = df[column_name]
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    return {'mean': mean,'median': median,'std_dev': std_dev}