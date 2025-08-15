import subprocess
import threading

def task_func(file_list):
    exit_codes = [0] * len(file_list)

    def run_file(file_path):
        result = subprocess.run(file_path, shell=True)
        exit_codes[file_list.index(file_path)] = result.returncode

    threads = [threading.Thread(target=run_file, args=(file,)) for file in file_list]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return exit_codes