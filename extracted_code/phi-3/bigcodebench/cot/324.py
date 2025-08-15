import subprocess
import threading

def run_file(file_path):
    try:
        result = subprocess.run(file_path, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode

def task_func(file_list):
    with threading.Semaphore(len(file_list)) as sem:
        exit_codes = []
        def worker(file_path):
            with sem:
                print(f"Running: {file_path}")
                exit_code = run_file(file_path)
                print(f"Finished: {file_path} with exit code {exit_code}")
                exit_codes.append(exit_code)

        threads = [threading.Thread(target=worker, args=(file_path,)) for file_path in file_list]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    return exit_codes