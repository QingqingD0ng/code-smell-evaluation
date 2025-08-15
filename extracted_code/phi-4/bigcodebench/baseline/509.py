import pandas as pd
import csv
from difflib import ndiff
import os

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    try:
        with open(file_path1, 'r', newline='') as f1, open(file_path2, 'r', newline='') as f2:
            reader1 = csv.reader(f1, delimiter=delimiter, quotechar=quotechar)
            reader2 = csv.reader(f2, delimiter=delimiter, quotechar=quotechar)
            
            lines1 = list(reader1)
            lines2 = list(reader2)

            if not lines1 or not lines2:
                raise ValueError("One or both files are empty.")

            diff = list(ndiff(lines1, lines2))
            result = []

            for i, line in enumerate(diff):
                if line.startswith(' '):
                    result.append({'Line Number': i + 1, 'Status':'', 'Content': line[2:]})
                elif line.startswith('-'):
                    result.append({'Line Number': i + 1, 'Status': '-', 'Content': line[2:]})
                elif line.startswith('+'):
                    result.append({'Line Number': i + 1, 'Status': '+', 'Content': line[2:]})

            return pd.DataFrame(result)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")