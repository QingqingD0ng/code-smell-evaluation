import pandas as pd
import os
import glob

def task_func(my_list, file_dir='./data_files/', file_ext='.csv'):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list.")
        
    my_list.append(12)
    num_files_to_concatenate = sum(my_list)
    
    files = glob.glob(os.path.join(file_dir, f"*{file_ext}"))
    
    if not files:
        raise FileNotFoundError("No files found in the specified directory.")
    
    files.sort()
    selected_files = files[:num_files_to_concatenate]
    
    dataframes = [pd.read_csv(file) for file in selected_files]
    result_df = pd.concat(dataframes, ignore_index=True)
    
    return result_df

def create_dummy_csv():
    os.makedirs('./data_files/', exist_ok=True)
    for i in range(10):
        df = pd.DataFrame({'A': range(5), 'B': range(3, 8)})
        df.to_csv(f'./data_files/dummy_{i}.csv', index=False)

def tearDown_dummy():
    files = glob.glob('./data_files/*.csv')
    for file in files:
        os.remove(file)
    os.rmdir('./data_files/')