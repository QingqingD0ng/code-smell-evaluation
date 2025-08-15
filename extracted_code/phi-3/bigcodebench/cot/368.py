import os
import shutil
import random

def task_func(src_dir, dest_dir, seed=100):
    random.seed(seed)
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    if not files:
        raise ValueError("No files found in the source directory.")
    file_to_move = random.choice(files)
    shutil.move(os.path.join(src_dir, file_to_move), dest_dir)
    return file_to_move