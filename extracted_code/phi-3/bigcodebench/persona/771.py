import re

import os

from pathlib import Path

import csv


def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:

    new_files = []

    path = Path(directory)

    for file_path in path.glob('*' + '.csv'):

        match = re.match(pattern, file_path.name)

        if match:

            base_name = match.group(1)

            new_file_name = f"{base_name}.csv"

            new_files.append(new_file_name)

            with open(file_path, 'r', newline='') as input_file:

                reader = csv.reader(input_file)

                lines = list(reader)

            with open(os.path.join(directory, new_file_name), 'w', newline='') as output_file:

                writer = csv.writer(output_file)

                writer.writerows(lines)


    return new_files