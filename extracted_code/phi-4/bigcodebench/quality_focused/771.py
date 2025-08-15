import re
import os
from pathlib import Path
import csv

def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:
    new_files = []
    dir_path = Path(directory)
    regex = re.compile(pattern)

    for file in dir_path.glob('*.csv'):
        match = regex.match(file.name)
        if match:
            new_name = match.group(1) + '.csv'
            new_file = dir_path / new_name
            with file.open('r', newline='') as f_read, new_file.open('w', newline='') as f_write:
                reader = csv.reader(f_read)
                writer = csv.writer(f_write)
                writer.writerows(reader)
            new_files.append(new_file.name)

    return new_files