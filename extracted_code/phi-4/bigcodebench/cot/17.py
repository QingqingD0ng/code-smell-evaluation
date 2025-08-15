import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == process_name:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
            action = "Restarting"
            break
    else:
        action = "Starting"

    subprocess.Popen([process_name])
    time.sleep(1)

    return f"Process not found. {action} {process_name}." if action == "Starting" else f"Process found. {action} {process_name}."