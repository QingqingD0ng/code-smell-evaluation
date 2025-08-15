import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    # Check for the process by name
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == process_name:
            # Terminate the found process
            proc.terminate()
            proc.wait()
            # Start the process again
            subprocess.Popen(process_name)
            return f"Process found. Restarting {process_name}."

    # If the process was not found, start it
    subprocess.Popen(process_name)
    return f"Process not found. Starting {process_name}."