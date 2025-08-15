import subprocess
import time
import threading

def task_func(file_list):
    exit_codes = [None] * len(file_list)
    threads = []

    def run_subprocess(file_path, index):
        try:
            result = subprocess.run(file_path, shell=True)
            exit_codes[index] = result.returncode
        except Exception as e:
            exit_codes[index] = e

    for i, file_path in enumerate(file_list):
        thread = threading.Thread(target=run_subprocess, args=(file_path, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return exit_codes

# Example usage:
# result = task_func(["task_func_data/file1.bat", "task_func_data/file2.bat"])
# print(result)