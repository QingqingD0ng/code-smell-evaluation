import subprocess
import os
import sys
import glob

def task_func(directory_path):
    bat_files = glob.glob(os.path.join(directory_path, '*.bat'))
    results = []
    for bat_file in bat_files:
        try:
            result = subprocess.run(bat_file, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results.append((os.path.basename(bat_file), result.returncode, result.stdout.decode(), result.stderr.decode()))
        except subprocess.CalledProcessError as e:
            results.append((os.path.basename(bat_file), None, e.stdout.decode(), e.stderr.decode()))
    return results

# The code has been optimized by removing unnecessary comments and refining the exception handling to include stdout and stderr outputs. This will provide more detailed information about the executed.bat files.