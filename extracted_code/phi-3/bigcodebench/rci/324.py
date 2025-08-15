import subprocess
import threading
from queue import Queue

def task_func(file_list):
    exit_codes = [0] * len(file_list)
    queue = Queue()
    threads = []

    def run_file(index, file_path):
        try:
            result = subprocess.run(file_path, shell=False)
            exit_codes[index] = result.returncode
        except Exception as e:
            exit_codes[index] = -1
            print(f"Error running file {file_path}: {str(e)}")

    for i, file in enumerate(file_list):
        queue.put((i, file))
        thread = threading.Thread(target=run_file, args=(i, file))
        threads.append(thread)
        thread.start()

    for _ in threads:
        queue.get()

    for thread in threads:
        thread.join()

    return exit_codes