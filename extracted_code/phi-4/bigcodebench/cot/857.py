import warnings
import os
import glob
import shutil

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    for extension in EXTENSIONS:
        pattern = os.path.join(SOURCE_DIR, f"*{extension}")
        for file_path in glob.glob(pattern):
            try:
                shutil.copy(file_path, DEST_DIR)
                transferred_files.append(os.path.basename(file_path))
            except Exception as e:
                warnings.warn(f"Could not transfer {file_path}: {e}")
    
    return transferred_files