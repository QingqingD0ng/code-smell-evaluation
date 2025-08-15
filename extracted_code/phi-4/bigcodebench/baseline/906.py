import zipfile
import os
import re
import shutil

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    archive_path = os.path.join(target_dir, archive_name)
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if re.search(r'_processed$', file):
                    full_file_path = os.path.join(root, file)
                    archive.write(full_file_path, os.path.relpath(full_file_path, source_dir))
    return archive_path