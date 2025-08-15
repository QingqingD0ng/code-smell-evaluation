import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    target_path = Path(target_dir)
    copied_files = []
    for file_path, content in kwargs.items():
        if content is not None:
            full_path = Path(file_path)
            if full_path.exists() and full_path.stat().st_size > 0:
                target = target_path / full_path.name
                shutil.copy(full_path, target)
                copied_files.append(str(target))
    return copied_files