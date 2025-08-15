import subprocess

import os

import sys

import glob


def task_func(directory_path):
    result = []
    for bat_file in glob.glob(os.path.join(directory_path, '*.bat')):
        try:
            with open(bat_file, 'r') as f:
                command = f.readline().strip()
            process = subprocess.run(command, shell=True, check=True)
            result.append((os.path.basename(bat_file), process.returncode))
        except (subprocess.CalledProcessError, FileNotFoundError):
            result.append((os.path.basename(bat_file), None))
    return result


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(task_func(directory))