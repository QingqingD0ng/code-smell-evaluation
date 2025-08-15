import os
import shutil
import random

def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory '{src_dir}' does not exist.")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    files = os.listdir(src_dir)
    if not files:
        raise ValueError("Source directory is empty.")
    
    random.seed(seed)
    selected_file = random.choice(files)
    src_file_path = os.path.join(src_dir, selected_file)
    dest_file_path = os.path.join(dest_dir, selected_file)
    
    try:
        shutil.move(src_file_path, dest_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to move file '{selected_file}': {e}")
    
    return selected_file