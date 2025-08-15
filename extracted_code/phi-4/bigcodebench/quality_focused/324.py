import subprocess
import threading

def task_func(file_list):
    exit_codes = [None] * len(file_list)
    threads = []

    def run_subprocess(index, file_name):
        result = subprocess.run(file_name, shell=True)
        exit_codes[index] = result.returncode

    for i, file_name in enumerate(file_list):
        thread = threading.Thread(target=run_subprocess, args=(i, file_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return exit_codes