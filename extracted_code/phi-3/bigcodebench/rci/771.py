import re

import os

from pathlib import Path

import csv

import logging


logging.basicConfig(level=logging.INFO)


def process_csv_file(file_path: Path, new_file_pattern: str) -> str:

    new_file_name = re.sub(new_file_pattern, r'\1.csv', file_path.name, 1)

    new_file_path = file_path.parent / new_file_name


    try:

        with open(file_path, 'r', newline='') as f_read, open(new_file_path, 'w', newline='') as f_write:

            reader = csv.reader(f_read)

            writer = csv.writer(f_write)

            for row in reader:

                writer.writerow(row)

        return new_file_name

    except Exception as e:

        logging.error(f'Error processing file {file_path}: {e}')

        return None


def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:

    new_files = []

    for file in Path(directory).glob('*.csv'):

        result = process_csv_file(file, pattern)

        if result:

            new_files.append(result)

    return new_files