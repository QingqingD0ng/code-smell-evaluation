import re
import os
from pathlib import Path
import csv

def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:
    new_files = []
    regex = re.compile(pattern)

    for filename in os.listdir(directory):
        match = regex.match(filename)
        if match:
            base_name = match.group(1)
            # Remove any trailing number from the base name
            new_base_name = re.sub(r'-\d+$', '', base_name)
            new_filename = f"{new_base_name}.csv"
            new_files.append(new_filename)
            # Copy content from the original file to the new file
            with open(os.path.join(directory, filename), 'r', newline='') as read_file:
                reader = csv.reader(read_file)
                with open(os.path.join(directory, new_filename), 'w', newline='') as write_file:
                    writer = csv.writer(write_file)
                    for row in reader:
                        writer.writerow(row)

    return new_files