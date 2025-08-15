import os
import shutil
import random

def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    random.seed(seed)
    files = os.listdir(src_dir)
    if not files:
        return ""
    file_to_move = random.choice(files)
    src_file_path = os.path.join(src_dir, file_to_move)
    dest_file_path = os.path.join(dest_dir, file_to_move)
    shutil.move(src_file_path, dest_file_path)
    return file_to_move