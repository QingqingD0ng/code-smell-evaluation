import pandas as pd
import os
import glob

def task_func(my_list, file_dir='./data_files/', file_ext='.csv'):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list")
    
    my_list.append(12)
    total_files_to_concat = sum(my_list)
    
    all_files = glob.glob(os.path.join(file_dir, '*' + file_ext))
    
    if not all_files:
        raise FileNotFoundError(f"No files found in {file_dir} with extension {file_ext}")
    
    dataframes = []
    for file in all_files[:total_files_to_concat]:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    result_df = pd.concat(dataframes, ignore_index=True)
    return result_df