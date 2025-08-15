import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    start_time = time.time()
    while True:
        try:
            if not any(proc.name() == process_name for proc in psutil.process_iter()):
                subprocess.Popen([process_name])
                return f"Process not found. Starting {process_name}."
            else:
                for proc in psutil.process_iter(['pid', 'name']):
                    if proc.info['name'] == process_name:
                        proc.kill()
                subprocess.Popen([process_name])
                return f"Process found. Restarting {process_name}."
        except Exception as e:
            return f"Error {e}. Retrying in 5 seconds..."
        time.sleep(5)
        if time.time() - start_time > 60:
            return "Process still not running after a minute, retrying..."