import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script {script_path} does not exist.")
    start_time = time.time()
    process = subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_id = process.pid
    try:
        while True:
            if time.time() - start_time > timeout:
                process.terminate()
                break
            try:
                process = psutil.Process(process_id)
                cpu_usage = process.cpu_percent()
                memory_usage = process.memory_info().rss
                break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                break
    except Exception as e:
        process.kill()
        raise e
    finally:
        process.wait()
    return {
        'CPU Usage': cpu_usage,
        'Memory Usage': memory_usage
    }