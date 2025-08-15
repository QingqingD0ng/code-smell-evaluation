import pandas as pd
import csv
from difflib import ndiff
import os

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    if not os.path.exists(file_path1) or not os.path.exists(file_path2):
        raise FileNotFoundError("One or both files not found.")

    with open(file_path1, 'r', newline='') as f1, open(file_path2, 'r', newline='') as f2:
        csv_reader1 = csv.reader(f1, delimiter=delimiter, quotechar=quotechar)
        csv_reader2 = csv_reader2 = csv.reader(f2, delimiter=delimiter, quotechar=quotechar)

        lines1 = [line for line in csv_reader1]
        lines2 = [line for line in csv_reader2]

    if not lines1 or not lines2:
        raise ValueError("One or both files are empty.")

    differences = []
    for index, (line1, line2) in enumerate(ndiff(lines1, lines2)):
        if line1.startswith('- '):
            differences.append((index + 1, '-', line2[2:]))
        elif line1.startswith('+ '):
            differences.append((index + 1, '+', line2[2:]))
        elif line1.startswith('  '):
            differences.append((index + 1,'', line1[2:]))

    df = pd.DataFrame(differences, columns=['Line Number', 'Status', 'Content'])
    return df