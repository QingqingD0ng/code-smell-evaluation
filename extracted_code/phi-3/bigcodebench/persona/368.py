import os

import shutil

import random


def task_func(src_dir: str, dest_dir: str, seed: int = 100) -> str:
    random.seed(seed)
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    file_to_move = random.choice(files)
    shutil.move(file_to_move, os.path.join(dest_dir, os.path.basename(file_to_move)))
    return os.path.basename(file_to_move)