import re

import os

from pathlib import Path

import csv


def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:

    new_files = []

    for file in Path(directory).glob('*.csv'):

        if re.match(pattern, file.name):

            new_file_name = re.sub(pattern, r'\1.csv', file.name, 1)

            new_file_path = file.parent / new_file_name

            with open(file, 'r') as f_read, open(new_file_path, 'w', newline='') as f_write:

                reader = csv.reader(f_read)

                writer = csv.writer(f_write)

                for row in reader:

                    writer.writerow(row)

            new_files.append(new_file_name)

    return new_files