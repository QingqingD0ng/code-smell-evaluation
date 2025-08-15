import os
import shutil
from typing import List

def task_func(src_dir: str, dest_dir: str, ext: str) -> List[str]:
    if not os.path.exists(src_dir) or not os.path.exists(dest_dir):
        raise FileNotFoundError(f"Source or destination directory does not exist: {src_dir} or {dest_dir}")

    moved_files = []
    for root, _, files in os.walk(src_dir):
        for filename in files:
            if filename.endswith(f'.{ext}'):
                file_path = os.path.join(root, filename)
                file_name = os.path.basename(file_path)
                dest_file_path = os.path.join(dest_dir, file_name)
                try:
                    # Check for file existence to prevent race conditions
                    if not os.path.exists(dest_file_path):
                        with open(file_path, 'rb') as f:
                            # Lock the destination file to prevent race conditions
                            f.seek(0)
                            dest_file_lock = open(dest_file_path, 'xb')
                            os.rename(dest_file_lock, dest_file_path)
                            shutil.move(file_path, dest_file_path)
                            moved_files.append(dest_file_path)
                except FileExistsError:
                    # Skip moving file if it already exists in the destination
                    pass
                except Exception as e:
                    # Handle other exceptions such as permission issues
                    print(f"An error occurred while moving file {filename}: {e}")

    return moved_files

# Example usage:
# test_src_dir = './test_src'
# test_dest_dir = './test_dest'
# test_ext = 'txt'
# os.makedirs(test_src_dir, exist_ok=True)
# os.makedirs(test_dest_dir, exist_ok=True)
# moved_files = task_func(test_src_dir, test_dest_dir, test