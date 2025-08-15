import subprocess
import os
import sys

def task_func(script_path, wait=True, *args, timeout=None):
    if not os.path.exists(script_path):
        raise ValueError("Script does not exist.")
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if wait:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                if process.returncode!= 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stderr)
                return process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(process.args, timeout, output=stdout, stderr=stderr)
        else:
            return None
    
    except Exception as e:
        raise RuntimeError(f"Error executing script: {e}")