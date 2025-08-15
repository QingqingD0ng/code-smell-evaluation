from pathlib import Path
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    file_path = Path(excel_file_path) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_excel(file_path)
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found")
    
    data = df[column_name]
    
    if not pd.api.types.is_numeric_dtype(data):
        raise ValueError(f"Column '{column_name}' contains non-numeric data")
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_dev_val = np.std(data)
    
    return {
       'mean': mean_val,
       'median': median_val,
       'std_dev': std_dev_val
    }