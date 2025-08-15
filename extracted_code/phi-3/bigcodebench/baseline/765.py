import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    target_path = Path(target_dir)
    copied_files = []

    for file_path, content in kwargs.items():
        if content is not None:
            path = Path(file_path)
            if path.exists() and path.stat().st_size > 0:
                target_file_path = target_path / path.name
                shutil.copy(file_path, target_file_path)
                copied_files.append(str(target_file_path))

    return copied_files