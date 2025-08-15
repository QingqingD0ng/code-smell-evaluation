import subprocess
import threading

def task_func(file_list):
    def run_subprocess(file, results, index):
        try:
            result = subprocess.run(file, shell=True, capture_output=True)
            results[index] = result.returncode
        except Exception as e:
            results[index] = f"Error: {e}"

    threads = []
    results = [None] * len(file_list)

    for index, file in enumerate(file_list):
        thread = threading.Thread(target=run_subprocess, args=(file, results, index))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results