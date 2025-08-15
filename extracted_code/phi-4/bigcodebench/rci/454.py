import os
import shutil

def task_func(src_dir, dest_dir, ext):
    if not isinstance(src_dir, str) or not isinstance(dest_dir, str) or not isinstance(ext, str):
        raise TypeError("All parameters must be strings")

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory '{src_dir}' does not exist")
    if not os.path.exists(dest_dir):
        raise FileNotFoundError(f"Destination directory '{dest_dir}' does not exist")

    moved_files = []
    ext = ext if ext.startswith('.') else f'.{ext}'

    for file_name in os.listdir(src_dir):
        if file_name.endswith(ext):
            file_path = os.path.join(src_dir, file_name)
            dest_file_path = os.path.join(dest_dir, file_name)

            if not os.path.exists(dest_file_path):
                try:
                    shutil.move(file_path, dest_file_path)
                    moved_files.append(dest_file_path)
                except Exception as e:
                    print(f"Error moving file {file_path} to {dest_file_path}: {e}")

    return moved_files