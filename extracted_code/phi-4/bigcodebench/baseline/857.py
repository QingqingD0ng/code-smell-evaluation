import warnings
import os
import glob
import shutil
import time

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    for ext in EXTENSIONS:
        for file_path in glob.glob(os.path.join(SOURCE_DIR, f"*{ext}")):
            try:
                shutil.copy2(file_path, DEST_DIR)
                transferred_files.append(os.path.basename(file_path))
            except Exception as e:
                warnings.warn(f"Failed to transfer {os.path.basename(file_path)}: {e}")
            time.sleep(0.1)  # Small delay to prevent rapid file system access
    return transferred_files