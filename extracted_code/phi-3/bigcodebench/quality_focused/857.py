import warnings
import os
import glob
import shutil
import time

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    for extension in EXTENSIONS:
        for filename in glob.glob(os.path.join(SOURCE_DIR, f'*{extension}')):
            try:
                shutil.copy2(filename, DEST_DIR)
                transferred_files.append(os.path.basename(filename))
            except Exception as e:
                warnings.warn(f'Could not transfer {filename}: {e}')
    return transferred_files