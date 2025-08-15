import subprocess
import os
import glob

def task_func(directory_path):
    if not os.path.isdir(directory_path):
        raise ValueError(f"Invalid directory path: {directory_path}")

    bat_files = glob.glob(os.path.join(directory_path, "*.bat"))
    results = []
    for bat_file in bat_files:
        try:
            result = subprocess.run([bat_file], capture_output=True, text=True)
            results.append((os.path.basename(bat_file), result.returncode))
        except FileNotFoundError:
            results.append((os.path.basename(bat_file), None))
        except subprocess.SubprocessError as e:
            results.append((os.path.basename(bat_file), None, str(e)))
    return results