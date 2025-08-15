import pandas as pd
from difflib import ndiff
import os

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    # Check if files exist
    if not os.path.exists(file_path1):
        raise FileNotFoundError(f"File not found: {file_path1}")
    if not os.path.exists(file_path2):
        raise FileNotFoundError(f"File not found: {file_path2}")

    # Read CSV files
    with open(file_path1, 'r', newline='', encoding='utf-8') as f1, open(file_path2, 'r', newline='', encoding='utf-8') as f2:
        reader1 = list(csv.reader(f1, delimiter=delimiter, quotechar=quotechar))
        reader2 = list(csv.reader(f2, delimiter=delimiter, quotechar=quotechar))

    # Check if files are empty
    if not reader1:
        raise ValueError(f"File is empty: {file_path1}")
    if not reader2:
        raise ValueError(f"File is empty: {file_path2}")

    # Perform line-by-line comparison
    diff = ndiff(reader1, reader2)

    # Prepare data for DataFrame
    data = []
    for i, line in enumerate(diff, start=1):
        status = line[0]
        content = line[2:]  # Remove the status character(s)
        content = ''.join(content).strip()  # Join and strip for clean content
        data.append((i, status, content))

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Line Number', 'Status', 'Content'])
    
    return df