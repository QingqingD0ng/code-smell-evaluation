import subprocess

import time

import threading


def task_func(file_list):
    exit_codes = []

    def run_file(file_name):
        result = subprocess.run(["cmd.exe", "/c", file_name], capture_output=True)
        exit_codes.append(result.returncode)

    threads = []
    for file_name in file_list:
        thread = threading.Thread(target=run_file, args=(file_name,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return exit_codes


# Example usage

exit_codes = task_func(["task_func_data/file1.bat", "task_func_data/file2.bat"])
print(exit_codes)