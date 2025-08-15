import subprocess
import os
import sys

def task_func(script_path, wait=True, *args):
    if not os.path.exists(script_path):
        raise ValueError("Script does not exist.")
    
    command = [sys.executable, script_path] + list(args)
    
    if wait:
        result = subprocess.run(command, check=True)
        return result.returncode
    else:
        subprocess.Popen(command)
        return None