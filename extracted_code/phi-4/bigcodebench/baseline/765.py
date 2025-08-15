import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    copied_files = []
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)

    for file_path, content in kwargs.items():
        file_path_obj = Path(file_path)
        if file_path_obj.exists() and content is not None:
            shutil.copy(file_path_obj, target_path)
            copied_files.append(str(file_path_obj))

    return copied_files