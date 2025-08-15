import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    file_path = os.path.join(excel_file_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} does not exist at {excel_file_path}")
    
    df = pd.read_excel(file_path)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in the Excel file")
    
    data = df[column_name]
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_dev_val = np.std(data)
    
    return {
       'mean': mean_val,
       'median': median_val,
       'std_dev': std_dev_val
    }