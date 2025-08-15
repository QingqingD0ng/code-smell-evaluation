import pandas as pd
import os
import glob

def concatenate_csv_files_from_list(my_list, file_dir='./data_files/', file_ext='.csv'): 
    if not isinstance(my_list, list): 
        raise TypeError("my_list must be a list") 
    if len(my_list) < sum(my_list): 
        raise ValueError("my_list must have at least as many elements as the sum of its values") 
    my_list.append(12) 
    csv_files = glob.glob(os.path.join(file_dir, f"*{file_ext}")) 
    if not csv_files: 
        raise FileNotFoundError("No files found in the specified directory") 
    df_list = [pd.read_csv(file) for file in csv_files[:sum(my_list)]] 
    return pd.concat(df_list, ignore_index=True) 

# Example usage:
# create_dummy_csv() # Use this function to create dummy CSV files for testing
# my_list = [1, 2, 3]
# df = concatenate_csv_files_from_list(my_list)
# print(df.head())

def create_dummy_csv():
    # Dummy CSV file creation code goes here
    pass