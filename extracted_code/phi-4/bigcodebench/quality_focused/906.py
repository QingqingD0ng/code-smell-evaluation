import os
import re
import shutil
import zipfile

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, archive_name)
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if '_processed' in file:
                    file_path = os.path.join(root, file)
                    archive.write(file_path, os.path.relpath(file_path, start=source_dir))
    
    return archive_path