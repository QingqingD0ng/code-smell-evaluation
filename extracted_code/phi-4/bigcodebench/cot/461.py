import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        return {'CPU Usage': 0, 'Memory Usage': 0}

    process = subprocess.Popen(["bash", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start_time = time.time()
    cpu_usage = 0
    memory_usage = 0

    try:
        while True:
            if process.poll() is not None:
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                process.terminate()
                break

            process_psutil = psutil.Process(process.pid)
            cpu_usage += process_psutil.cpu_percent(interval=0.1)
            memory_usage += process_psutil.memory_info().rss

            time.sleep(0.1)
    except psutil.NoSuchProcess:
        pass

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}