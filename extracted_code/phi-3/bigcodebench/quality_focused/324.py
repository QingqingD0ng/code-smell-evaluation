import subprocess

import time

import threading


def task_func(file_list):
    exit_codes = []

    def run_file(file_path):
        process = subprocess.Popen(file_path, shell=True)
        exit_codes.append(process.wait())

    threads = [threading.Thread(target=run_file, args=(file,)) for file in file_list]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return exit_codes


# Example usage:

exit_codes = task_func(["task_func_data/file1.bat", "task_func_data/file2.bat"])

print(exit_codes)