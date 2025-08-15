import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    full_path = os.path.join(excel_file_path, file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No file found at {full_path}")
    
    df = pd.read_excel(full_path)
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel file")
    
    data = df[column_name]
    mean_value = np.mean(data)
    median_value = np.median(data)
    std_dev = np.std(data)
    
    return {
      'mean': mean_value,
      'median': median_value,
      'std_dev': std_dev
    }