import warnings
import os
import glob
import shutil
import time

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    os.makedirs(DEST_DIR, exist_ok=True)
    
    for ext in EXTENSIONS:
        for file_path in glob.glob(os.path.join(SOURCE_DIR, f"*{ext}")):
            try:
                shutil.copy(file_path, DEST_DIR)
                transferred_files.append(os.path.basename(file_path))
            except Exception as e:
                warnings.warn(f"Failed to transfer {file_path}: {e}")
    
    time.sleep(0.1)  # Simulate processing time
    return transferred_files