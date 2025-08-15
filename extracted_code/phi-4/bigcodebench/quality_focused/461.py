import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        return {'CPU Usage': 0, 'Memory Usage': 0}

    process = subprocess.Popen(['bash', script_path])
    start_time = time.time()
    cpu_usage = 0
    memory_usage = 0

    try:
        while process.poll() is None:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout:
                process.terminate()
                break

            try:
                proc = psutil.Process(process.pid)
                cpu_usage += proc.cpu_percent(interval=0.1) / 10
                memory_usage += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(0.1)

    finally:
        process.terminate()
        process.wait()

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}