import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    restarted = False

    for _ in range(3):  # Attempt to restart the process up to 3 times
        if psutil.pid_exists(get_process_id(process_name)):
            restarted = True
            terminate_process(process_name)
            time.sleep(2)  # Wait for 2 seconds before restarting
        else:
            restarted = False
            break  # Exit the loop if process is not running

    if not restarted:
        return f"Process not found. Starting {process_name}."
    else:
        return f"Process found. Restarting {process_name}."

def get_process_id(process_name: str) -> int:
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None

def terminate_process(process_name: str) -> None:
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            proc.terminate()
            break

# Example usage:
print(task_func('notepad'))