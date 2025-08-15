import warnings
import os
import glob
import shutil
import time

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    for ext in EXTENSIONS:
        for file_path in glob.glob(os.path.join(SOURCE_DIR, f'*{ext}')):
            try:
                shutil.move(file_path, DEST_DIR)
                transferred_files.append(os.path.basename(file_path))
            except Exception as e:
                warnings.warn(f"Could not transfer {file_path}: {e}")
    return transferred_files