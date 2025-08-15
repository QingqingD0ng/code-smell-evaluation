import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    def is_process_running(name: str) -> bool:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == name:
                return True
        return False

    def terminate_and_restart(name: str):
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == name:
                proc.terminate()
                proc.wait()
        subprocess.Popen(name)

    if not is_process_running(process_name):
        subprocess.Popen(process_name)
        return f"Process not found. Starting {process_name}."
    else:
        terminate_and_restart(process_name)
        return f"Process found. Restarting {process_name}."