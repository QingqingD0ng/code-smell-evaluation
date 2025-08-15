import warnings
import os
import glob
import shutil
import time

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    for ext in EXTENSIONS:
        for filepath in glob.glob(os.path.join(SOURCE_DIR, f'*{ext}')):
            filename = os.path.basename(filepath)
            try:
                shutil.move(filepath, os.path.join(DEST_DIR, filename))
                transferred_files.append(filename)
            except Exception as e:
                warnings.warn(f'Could not transfer {filename}: {e}')
    return transferred_files