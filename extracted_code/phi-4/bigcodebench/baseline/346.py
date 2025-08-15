import subprocess
import os
import sys
import time

def task_func(script_path, wait=True, *args):
    if not os.path.exists(script_path):
        raise ValueError("Script does not exist.")
    
    process = subprocess.Popen([sys.executable, script_path] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if wait:
        stdout, stderr = process.communicate()
        if process.returncode!= 0:
            raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stderr)
        return process.returncode
    else:
        return None