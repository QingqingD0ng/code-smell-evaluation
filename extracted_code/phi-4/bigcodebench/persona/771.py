import re
import os
from pathlib import Path
import csv

def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:
    new_files = []
    path = Path(directory)

    for file in path.glob('*.csv'):
        match = re.match(pattern, file.name)
        if match:
            new_filename = f"{match.group(1)}.csv"
            new_file_path = path / new_filename

            with open(file, 'r', newline='') as infile:
                reader = csv.reader(infile)
                with open(new_file_path, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(reader)

            new_files.append(new_file_path.name)

    return new_files