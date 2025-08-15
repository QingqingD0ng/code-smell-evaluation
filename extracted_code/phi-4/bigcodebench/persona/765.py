import os
from pathlib import Path
import shutil

def task_func(kwargs, target_dir="non_none_files"):
    copied_files = []
    target_path = Path(target_dir)

    # Ensure the target directory exists
    target_path.mkdir(exist_ok=True)

    for file_path, content in kwargs.items():
        file_path_obj = Path(file_path)
        
        # Check if the file exists
        if file_path_obj.exists():
            # Check if the content is not None
            if content is not None:
                # Construct the destination path
                dest_path = target_path / file_path_obj.name
                
                # Copy the file
                shutil.copy(file_path, dest_path)
                copied_files.append(str(dest_path))
    
    return copied_files