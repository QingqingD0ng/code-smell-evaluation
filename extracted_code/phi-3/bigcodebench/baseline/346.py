import subprocess
import os
import sys
import time

def task_func(script_path, wait=True, *args):
    if not os.path.isfile(script_path):
        raise ValueError("The script does not exist.")
    try:
        result = subprocess.run([sys.executable, script_path] + list(args), check=True, text=True, capture_output=True)
        if wait:
            return result.returncode
        else:
            return None
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(e.returncode, e.cmd) from e