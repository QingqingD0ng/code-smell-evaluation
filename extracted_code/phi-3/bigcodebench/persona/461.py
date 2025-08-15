import subprocess
import psutil
import time
import os


def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        raise ValueError("Script path does not exist.")

    start_time = time.time()
    process = subprocess.Popen(["bash", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_id = process.pid

    cpu_usage = 0
    memory_usage = 0
    try:
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.kill()
                break
            try:
                cpu = psutil.cpu_percent(interval=1)
                memory_info = process.memory_info()
                cpu_usage += cpu
                memory_usage += memory_info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process.kill()
                break
            time.sleep(0.1)
    except psutil.AccessDenied:
        raise PermissionError("Permission denied to access process information.")
    except psutil.NoSuchProcess:
        raise ValueError("Process no longer exists.")

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}


# Example usage:
# resources = task_func('/path/to/script.sh')
# print(resources)