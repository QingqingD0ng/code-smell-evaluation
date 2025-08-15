import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    try:
        if psutil.process_list(proc=psutil.Process(subprocess.Popen([process_name]).pid)):
            return f"Process found. Restarting {process_name}."
        else:
            return f"Process not found. Starting {process_name}."
    except psutil.NoSuchProcess:
        return "Process not found. Starting {}.".format(process_name)
    except subprocess.CalledProcessError:
        return "Failed to start the process. Please check the command."