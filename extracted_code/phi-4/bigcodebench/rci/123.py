import pandas as pd
import os
import glob

def task_func(my_list, file_dir='./data_files/', file_ext='.csv'):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list")
    
    if not all(isinstance(x, (int, float)) for x in my_list):
        raise ValueError("All elements in my_list must be numbers")
    
    ELEMENT_TO_ADD = 12
    modified_list = my_list.copy()
    modified_list.append(ELEMENT_TO_ADD)
    
    total_files_to_concat = sum(modified_list)
    
    csv_files = glob.glob(os.path.join(file_dir, '*' + file_ext))
    
    if not csv_files:
        raise FileNotFoundError(f"No files found in {file_dir} with extension {file_ext}")
    
    dataframes = []
    for file in csv_files[:total_files_to_concat]:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except pd.errors.EmptyDataError:
            continue
    
    if not dataframes:
        raise FileNotFoundError("No valid CSV files found for concatenation.")
    
    result_df = pd.concat(dataframes, ignore_index=True)
    return result_df