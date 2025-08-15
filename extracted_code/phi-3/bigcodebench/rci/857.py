import warnings
import os
import glob
import shutil
from pathlib import Path

def task_func(SOURCE_DIR, DEST_DIR, EXTENSIONS):
    transferred_files = []
    
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)

    if not source_path.exists() or not source_path.is_dir():
        warnings.warn(f"Source directory {SOURCE_DIR} does not exist or is not a directory.")
        return transferred_files
    
    if not dest_path.exists() or not dest_path.is_dir():
        warnings.warn(f"Destination directory {DEST_DIR} does not exist or is not a directory.")
        return transferred_files
    
    if not os.access(dest_path, os.W_OK):
        warnings.warn(f"Destination directory {DEST_DIR} is not writable.")
        return transferred_files
    
    for ext in EXTENSIONS:
        for filepath in source_path.glob(f'*{ext}'):
            filename = filepath.name
            dest_filepath = dest_path / filename

            try:
                shutil.move(str(filepath), str(dest_filepath))
                transferred_files.append(filename)
            except FileExistsError:
                warnings.warn(f'File {filename} already exists in the destination directory. Skipping.')

    return transferred_files