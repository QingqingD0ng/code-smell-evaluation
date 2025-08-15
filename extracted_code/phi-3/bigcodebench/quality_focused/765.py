import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists
    copied_files = []
    
    for file_path, content in kwargs.items():
        if content and os.path.isfile(file_path):
            target_file_path = target_path / Path(file_path).name
            shutil.copy(file_path, target_file_path)
            copied_files.append(str(target_file_path))
    
    return copied_files