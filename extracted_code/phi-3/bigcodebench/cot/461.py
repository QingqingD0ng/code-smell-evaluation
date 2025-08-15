import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script {script_path} does not exist")
    
    process = subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start_time = time.time()
    cpu_usage = 0
    memory_usage = 0

    while process.poll() is None:
        if time.time() - start_time > timeout:
            process.kill()
            break
        time.sleep(0.1)
        try:
            process_info = psutil.Process(process.pid)
            cpu_usage += process_info.cpu_percent()
            memory_usage += process_info.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process.kill()
            raise
    process.wait()

    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}