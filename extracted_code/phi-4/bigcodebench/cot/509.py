import pandas as pd
import csv
from difflib import ndiff
import os

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    if not os.path.exists(file_path1) or not os.path.exists(file_path2):
        raise FileNotFoundError("One or both files not found.")

    with open(file_path1, 'r', newline='') as f1, open(file_path2, 'r', newline='') as f2:
        reader1 = csv.reader(f1, delimiter=delimiter, quotechar=quotechar)
        reader2 = csv.reader(f2, delimiter=delimiter, quotechar=quotechar)

        lines1 = list(reader1)
        lines2 = list(reader2)

    if not lines1 or not lines2:
        raise ValueError("One or both files are empty.")

    diff = list(ndiff([str(row) for row in lines1], [str(row) for row in lines2]))

    data = []
    for i, line in enumerate(diff, start=1):
        status = line[0]
        content = line[2:] if len(line) > 2 else ''
        data.append({'Line Number': i, 'Status': status, 'Content': content})

    return pd.DataFrame(data)