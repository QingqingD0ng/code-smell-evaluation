import subprocess
import os
import sys
import time

def task_func(script_path, wait=True, *args):
    if not os.path.isfile(script_path):
        raise ValueError(f"Script {script_path} does not exist.")

    try:
        process = subprocess.Popen([sys.executable, script_path] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if wait:
            stdout, stderr = process.communicate()
            return process.returncode
        else:
            return None
    except subprocess.CalledProcessError as e:
        raise e