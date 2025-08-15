import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10, interval=0.1) -> dict:
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script path does not exist: {script_path}")

    process = subprocess.Popen(['bash', script_path])
    start_time = time.time()
    pid = process.pid
    proc = psutil.Process(pid)

    cpu_usage = 0.0
    memory_usage = 0

    while process.poll() is None:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            process.kill()
            break

        try:
            cpu_usage += proc.cpu_percent(interval=min(interval, timeout - elapsed))
            memory_usage += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}