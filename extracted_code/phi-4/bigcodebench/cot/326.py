import subprocess
import os
import glob

def task_func(directory_path):
    result = []
    for bat_file in glob.glob(os.path.join(directory_path, "*.bat")):
        try:
            completed_process = subprocess.run([bat_file], capture_output=True)
            result.append((os.path.basename(bat_file), completed_process.returncode))
        except Exception:
            result.append((os.path.basename(bat_file), None))
    return result