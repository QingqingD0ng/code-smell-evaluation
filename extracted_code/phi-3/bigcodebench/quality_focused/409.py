import os
import pandas as pd
import numpy as np

def task_func(excel_file_path, file_name, column_name):
    # Construct file path
    file_path = os.path.join(excel_file_path, file_name)
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the directory {excel_file_path}.")
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    
    # Check if the specified column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel file.")
    
    # Calculate mean, median, and standard deviation
    mean = df[column_name].mean()
    median = df[column_name].median()
    std_dev = df[column_name].std()
    
    return {
       'mean': mean,
       'median': median,
       'std_dev': std_dev
    }

# Example usage (uncomment to run):
# data_dir, file_name = './data/', 'excel_file1.xls'
# stats = task_func(data_dir, file_name, 'Sales')
# os.remove(data_dir + file_name)
# os.rmdir(data_dir)
# print(stats)