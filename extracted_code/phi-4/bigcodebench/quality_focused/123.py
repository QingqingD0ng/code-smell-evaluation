import pandas as pd
import os
import glob

def task_func(my_list, file_dir='./data_files/', file_ext='.csv'):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list")

    my_list.append(12)
    num_files = sum(my_list)

    pattern = os.path.join(file_dir, f"*{file_ext}")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("No files found in the specified directory")

    files_to_concatenate = files[:num_files]
    df = pd.concat((pd.read_csv(f) for f in files_to_concatenate), ignore_index=True)

    return df