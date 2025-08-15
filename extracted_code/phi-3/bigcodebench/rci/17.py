import subprocess
import psutil
import time

def start_process(process_name: str) -> str:
    try:
        subprocess.run(["cmd", "/c", process_name], check=True)
        return f"Process started successfully."
    except FileNotFoundError:
        return f"Process not found. Attempting to start {process_name}."
    except subprocess.CalledProcessError as e:
        return str(e)
    except Exception as e:
        return str(e)

def monitor_and_restart_process(process_name: str, restart_delay: int = 5) -> str:
    """
    Monitor the specified process and restart it if it is not running.

    Parameters:
    - process_name (str): The name of the process to monitor and manage.
    - restart_delay (int): The delay (in seconds) before restarting the process if it is not found.

    Returns:
    - str: A message indicating the action taken.
    """
    while True:
        if psutil.pid_exists(start_process(process_name).split(" ")[-1]):
            return f"Process found. No need to restart."
        else:
            time.sleep(restart_delay)
            return f"Process not found. Attempting to start {process_name}."