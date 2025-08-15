import subprocess
import os
import sys
import glob

def task_func(directory_path):
    batch_files = glob.glob(os.path.join(directory_path, '*.bat'))
    results = []

    for batch_file in batch_files:
        try:
            result = subprocess.run(batch_file, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results.append((os.path.basename(batch_file), result.returncode))
        except subprocess.CalledProcessError as e:
            results.append((os.path.basename(batch_file), None))

    return results

if __name__ == "__main__":
    directory_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(task_func(directory_path))