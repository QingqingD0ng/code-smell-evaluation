import re
import os
from pathlib import Path
import csv

def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:
    new_files = []
    path_obj = Path(directory)

    if path_obj.is_dir():
        for file_path in path_obj.glob(pattern):
            base_name = re.match(pattern, file_path.name).group(1)
            new_file_name = base_name + ".csv"
            new_file_path = path_obj / new_file_name
            with open(file_path, 'r') as infile, open(new_file_path, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                for row in reader:
                    writer.writerow(row)
            new_files.append(new_file_path.name)
    return new_files