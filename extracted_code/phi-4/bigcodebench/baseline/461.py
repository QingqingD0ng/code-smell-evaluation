import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        return {'CPU Usage': 0, 'Memory Usage': 0}

    process = subprocess.Popen(['bash', script_path])
    start_time = time.time()
    pid = process.pid
    proc = psutil.Process(pid)

    cpu_usage = 0.0
    memory_usage = 0

    try:
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.kill()
                break

            cpu_usage += proc.cpu_percent(interval=0.1)
            memory_usage += proc.memory_info().rss

            if process.poll() is not None:
                break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}