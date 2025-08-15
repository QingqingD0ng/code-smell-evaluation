import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    path_target = Path(target_dir)
    if not path_target.exists():
        path_target.mkdir(parents=True)

    copied_files = []
    for file_path, content in kwargs.items():
        file_path = Path(file_path)
        if file_path.is_file() and content:
            destination = path_target / file_path.name
            shutil.copy2(file_path, destination)
            copied_files.append(str(destination.resolve()))
    
    return copied_files