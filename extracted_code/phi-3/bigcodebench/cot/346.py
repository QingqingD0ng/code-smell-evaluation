import subprocess
import os
import sys
import time

def task_func(script_path, wait=True, *args):
    if not os.path.isfile(script_path):
        raise ValueError("The script does not exist.")
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if wait:
            return process.wait()
        else:
            return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(e.returncode)