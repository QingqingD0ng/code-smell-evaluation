import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"The script {script_path} does not exist.")

    start_time = time.time()
    process = subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:
        if time.time() - start_time > timeout:
            process.kill()
            break
        try:
            psutil.Process(process.pid).wait(timeout=1)
        except psutil.NoSuchProcess:
            break

    cpu_usage = psutil.Process(process.pid).cpu_percent()
    memory_usage = psutil.Process(process.pid).memory_info().rss

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}

# Example usage:
# resources = task_func('/path/to/script.sh')
# print(resources)