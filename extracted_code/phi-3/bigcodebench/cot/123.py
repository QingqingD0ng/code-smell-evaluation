import pandas as pd
import os
import glob

def task_func(my_list, file_dir='./data_files/', file_ext='.csv'):
    if not isinstance(my_list, list):
        raise TypeError('my_list must be a list')
    
    my_list.append(12)
    
    files = sorted(glob.glob(os.path.join(file_dir, f'*{file_ext}')))
    if not files:
        raise FileNotFoundError('No files found in the specified directory')
    
    df_list = [pd.read_csv(file) for file in files[:len(my_list)]]
    
    return pd.concat(df_list, ignore_index=True)
When using the function `task_func`, ensure that you provide a list as the first argument. The second argument `file_dir` defaults to './data_files/' if not specified. The third argument `file_ext` defaults to '.csv' if not specified. The function modifies the list by appending '12' and concatenates CSV files according to the modified list length. It raises a `TypeError` if the first argument is not a list and a `FileNotFoundError` if no files are found in the specified directory.