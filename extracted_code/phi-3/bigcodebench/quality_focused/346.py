import subprocess
import os

def task_func(script_path, wait=True, *args):
    if not os.path.exists(script_path):
        raise ValueError("Script does not exist")
    
    try:
        result = subprocess.run([sys.executable, script_path] + list(args), check=True)
        return result.returncode if wait else None
    except subprocess.CalledProcessError as e:
        raise e
    except Exception as e:
        raise e