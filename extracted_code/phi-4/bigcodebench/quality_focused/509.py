import pandas as pd
import csv
from difflib import ndiff
import os

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    if not os.path.exists(file_path1) or not os.path.exists(file_path2):
        raise FileNotFoundError("One or both files not found.")

    with open(file_path1, mode='r', newline='') as file1, open(file_path2, mode='r', newline='') as file2:
        reader1 = csv.reader(file1, delimiter=delimiter, quotechar=quotechar)
        reader2 = csv.reader(file2, delimiter=delimiter, quotechar=quotechar)

        lines1 = list(reader1)
        lines2 = list(reader2)

    if not lines1 or not lines2:
        raise ValueError("One or both files are empty.")

    diff = ndiff(lines1, lines2)

    data = []
    for i, s in enumerate(diff, 1):
        status = s[0]
        content = ''.join(s[2:]) if len(s) > 2 else ''
        data.append((i, status, content))

    df = pd.DataFrame(data, columns=['Line Number', 'Status', 'Content'])
    return df