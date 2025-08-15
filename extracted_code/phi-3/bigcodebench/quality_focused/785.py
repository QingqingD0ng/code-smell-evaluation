import subprocess

import os

import glob


ARCHIVE_DIR = '/tmp/archive'


def task_func(pattern):
    file_paths = glob.glob(pattern)
    archive_name = os.path.join(ARCHIVE_DIR, os.path.basename(pattern))
    with open(archive_name, 'wb') as archive_file:
        for file_path in file_paths:
            subprocess.run(['tar', 'czf', archive_name, '-T', '-', file_path])
    for file_path in file_paths:
        os.remove(file_path)
    return archive_name


# Example usage:

# archive_path = task_func('*.txt')