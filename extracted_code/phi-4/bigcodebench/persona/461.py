import subprocess
import psutil
import time
import os

def task_func(script_path: str, timeout=10) -> dict:
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script path {script_path} does not exist.")

    process = subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps_process = psutil.Process(process.pid)

    start_time = time.time()
    cpu_usage = 0.0
    memory_usage = 0

    try:
        while True:
            if ps_process.is_running():
                cpu_usage += ps_process.cpu_percent(interval=0.1)
                memory_info = ps_process.memory_info()
                memory_usage += memory_info.rss

                if time.time() - start_time > timeout:
                    process.terminate()
                    raise TimeoutError("Process terminated due to timeout.")

                time.sleep(0.1)
            else:
                break
    except psutil.NoSuchProcess:
        pass

    process.communicate()
    return {'CPU Usage': cpu_usage, 'Memory Usage': memory_usage}