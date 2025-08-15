import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    file_path = os.path.join(excel_file_path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_excel(file_path)
    
    if column_name not in df.columns:
        raise ValueError(f"The column {column_name} is not found in the Excel file.")
    
    data = df[column_name].dropna()
    
    mean_value = np.mean(data)
    median_value = np.median(data)
    std_dev_value = np.std(data, ddof=0)
    
    return {
       'mean': mean_value,
       'median': median_value,
       'std_dev': std_dev_value
    }