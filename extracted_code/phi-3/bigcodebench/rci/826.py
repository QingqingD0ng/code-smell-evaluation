import os
import shutil
import re

def task_func(source_dir, target_dir, file_pattern=r'\b[A-Za-z0-9]+\.(txt|doc|docx)\b'):
    moved_files_count = 0
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in os.listdir(source_dir):
        if re.match(file_pattern, filename):
            src_file_path = os.path.join(source_dir, filename)
            dst_file_path = os.path.join(target_dir, filename)
            
            try:
                shutil.move(src_file_path, dst_file_path)
                moved_files_count += 1
                print(f'Moved {filename} to {target_dir}')
            except Exception as e:
                print(f'Error moving {filename}: {str(e)}')
    
    return moved_files_count