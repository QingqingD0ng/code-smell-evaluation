import os
import shutil
import random

def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    random.seed(seed)
    files = os.listdir(src_dir)
    if not files:
        raise FileNotFoundError("Source directory is empty.")
    
    selected_file = random.choice(files)
    src_file_path = os.path.join(src_dir, selected_file)
    dest_file_path = os.path.join(dest_dir, selected_file)
    
    shutil.move(src_file_path, dest_file_path)
    return selected_file