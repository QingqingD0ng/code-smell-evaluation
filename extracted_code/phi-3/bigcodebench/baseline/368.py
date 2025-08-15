import os
import shutil
import random

def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    random.seed(seed)
    files = os.listdir(src_dir)
    if files:
        filename = random.choice(files)
        source_file = os.path.join(src_dir, filename)
        shutil.move(source_file, dest_dir)
        return filename
    else:
        raise ValueError("Source directory is empty.")