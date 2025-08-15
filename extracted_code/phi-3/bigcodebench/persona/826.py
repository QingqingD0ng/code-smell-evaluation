import re
import os
import shutil

def task_func(source_dir, target_dir, file_pattern=r'\b[A-Za-z0-9]+\.(txt|doc|docx)\b'):
    moved_files_count = 0
    for filename in os.listdir(source_dir):
        if re.match(file_pattern, filename):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.move(source_file, target_file)
            moved_files_count += 1
    return moved_files_count

# Usage example
# moved_files = task_func('/path/to/source', '/path/to/target')
# print(f'{moved_files} files were moved.')