import subprocess
import os
import glob

def task_func(directory_path):
    bat_files = glob.glob(os.path.join(directory_path, '*.bat'))
    results = []
    for bat_file in bat_files:
        try:
            result = subprocess.run(bat_file, shell=True, check=False)
            results.append((os.path.basename(bat_file), result.returncode))
        except Exception:
            results.append((os.path.basename(bat_file), None))
    return results