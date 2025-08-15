import pandas as pd
import csv
from difflib import ndiff
import os

def get_differences(file1_lines, file2_lines):
    diff = ndiff(file1_lines, file2_lines)
    return [line for line in diff if line.startswith(('- ', '+ '))]

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    if not os.path.exists(file_path1) or not os.path.exists(file_path2):
        raise FileNotFoundError("One or both of the files cannot be found.")
    if os.path.getsize(file_path1) == 0 or os.path.getsize(file_path2) == 0:
        raise ValueError("One of the files is empty.")
    try:
        with open(file_path1, 'r', newline='', encoding='utf-8') as file1, open(file_path2, 'r', newline='', encoding='utf-8') as file2:
            reader1 = csv.reader(file1, delimiter=delimiter, quotechar=quotechar)
            reader2 = csv.reader(file2, delimiter=delimiter, quotechar=quotechar)
            file1_lines = [line for line in reader1]
            file2_lines = [line for line in reader2]
        differences = get_differences(file1_lines, file2_lines)
        df = pd.DataFrame(differences, columns=['Line Number', 'Status', 'Content'])
        df['Line Number'] = df['Line Number'].astype(int)
        return df
    except IOError as e:
        raise Exception("An IO error occurred while processing the files.") from e

# Example usage:
# df = task_func('file1.csv', 'file2.csv')
# os.remove('file1.csv')
# os.remove('file2.csv')
# print(df.head())