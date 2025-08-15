import os
import shutil
import random

def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    random.seed(seed)
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    if files:
        filename = random.choice(files)
        shutil.move(os.path.join(src_dir, filename), dest_dir)
        return filename
    else:
        raise FileNotFoundError("No files to move in the source directory.")