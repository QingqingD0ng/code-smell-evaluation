import os
import shutil
import random
from typing import Optional

def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    random.seed(seed)
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    if not files:
        raise FileNotFoundError("No files found in the source directory.")
    
    selected_file = random.choice(files)
    src_file_path = os.path.join(src_dir, selected_file)
    dest_file_path = os.path.join(dest_dir, selected_file)
    
    shutil.move(src_file_path, dest_file_path)
    
    return selected_file