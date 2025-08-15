import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    process = next((proc for proc in psutil.process_iter(['name']) if proc.info['name'] == process_name), None)

    if process:
        try:
            process.terminate()
            process.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass

    subprocess.Popen(process_name)
    return f"Process {'found. Restarting' if process else 'not found. Starting'} {process_name}."