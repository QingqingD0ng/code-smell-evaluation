import subprocess
import threading

def task_func(file_list):
    def run_subprocess(file):
        result = subprocess.run(file, shell=True)
        results.append(result.returncode)

    threads = []
    results = []

    for file in file_list:
        thread = threading.Thread(target=run_subprocess, args=(file,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results