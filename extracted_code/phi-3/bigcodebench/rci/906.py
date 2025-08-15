import os
import re
import shutil
import zipfile

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip', suffix: str = '_processed') -> str:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.access(target_dir, os.W_OK):
        raise PermissionError(f"Cannot write to target directory: {target_dir}")

    archive_path = os.path.join(target_dir, archive_name)
    file_regex = re.compile(rf"{re.escape(suffix)}$")

    try:
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file_regex.search(file):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, source_dir))
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating archive: {e}")

    return archive_path