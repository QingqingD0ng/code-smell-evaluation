import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    file_path = os.path.join(excel_file_path, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'Excel file not found: {file_path}')
    
    df = pd.read_excel(file_path)
    if column_name not in df.columns:
        raise ValueError(f'Column not found: {column_name}')
    
    data = df[column_name].dropna()
    return {
      'mean': data.mean(),
      'median': data.median(),
      'std_dev': data.std()
    }