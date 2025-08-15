import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    Path(target_dir).mkdir(exist_ok=True)
    copied_files = []
    for file_path, content in kwargs.items():
        if content is not None:
            if Path(file_path).exists():
                shutil.copy(file_path, target_dir)
                copied_files.append(file_path)
    return copied_files