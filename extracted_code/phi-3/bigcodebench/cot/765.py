import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    copied_files = []
    for file_path, content in kwargs.items():
        if content is not None and os.path.isfile(file_path):
            target_file_path = os.path.join(target_dir, os.path.basename(file_path))
            with open(file_path, 'w') as f_source, open(target_file_path, 'w') as f_target:
                f_source.write(content)
                f_target.write(content)
            shutil.copystat(file_path, target_file_path)
            copied_files.append(target_file_path)
    return copied_files