import re
import os
import shutil

def task_func(source_dir, target_dir, file_pattern=r'\b[A-Za-z0-9]+\.(txt|doc|docx)\b'):
    moved_files_count = 0
    file_regex = re.compile(file_pattern)
    
    for filename in os.listdir(source_dir):
        if file_regex.match(filename):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.move(source_path, target_path)
            moved_files_count += 1

    return moved_files_count