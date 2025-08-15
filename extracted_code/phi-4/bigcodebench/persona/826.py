import re
import os
import shutil

def task_func(source_dir, target_dir, file_pattern=r'\b[A-Za-z0-9]+\.(txt|doc|docx)\b'):
    moved_files_count = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(source_dir):
        if re.match(file_pattern, filename):
            src_file = os.path.join(source_dir, filename)
            tgt_file = os.path.join(target_dir, filename)
            shutil.move(src_file, tgt_file)
            moved_files_count += 1
    return moved_files_count