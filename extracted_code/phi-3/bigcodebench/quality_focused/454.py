import os
import shutil
import glob

def task_func(src_dir, dest_dir, ext):
    if not os.path.exists(src_dir) or not os.path.exists(dest_dir):
        raise FileNotFoundError("Source or destination directory does not exist")

    moved_files = []
    for file_path in glob.glob(os.path.join(src_dir, f"*.{ext}")):
        file_name = os.path.basename(file_path)
        dest_file_path = os.path.join(dest_dir, file_name)
        if not os.path.exists(dest_file_path):
            shutil.move(file_path, dest_file_path)
            moved_files.append(dest_file_path)
    
    return moved_files