import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    def is_process_running(name):
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == name:
                return True
        return False

    if is_process_running(process_name):
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == process_name:
                proc.terminate()
                proc.wait()
                time.sleep(1)
                subprocess.Popen(process_name)
                return f"Process found. Restarting {process_name}."
    else:
        subprocess.Popen(process_name)
        return f"Process not found. Starting {process_name}."