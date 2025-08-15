import re
import os
from pathlib import Path
import csv

def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:
    new_files = []
    for file in os.listdir(directory):
        if re.match(pattern, file):
            match = re.match(pattern, file)
            new_filename = match.group(1) + '.csv'
            new_filepath = Path(directory) / new_filename
            old_filepath = Path(directory) / file
            with open(old_filepath, 'r', newline='') as infile:
                reader = csv.reader(infile)
                with open(new_filepath, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    for row in reader:
                        writer.writerow(row)
            new_files.append(new_filename)
    return new_files