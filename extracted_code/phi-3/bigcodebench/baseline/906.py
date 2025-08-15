import zipfile
import os
import re
import shutil

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    with zipfile.ZipFile(os.path.join(target_dir, archive_name), 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if re.search(r'_processed$', file):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, source_dir))
    return os.path.join(target_dir, archive_name)