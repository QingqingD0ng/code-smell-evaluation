import zipfile
import os
import re
import shutil

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    processed_files = [f for f in os.listdir(source_dir) if re.search(r'_processed$', f)]
    archive_path = os.path.join(target_dir, archive_name)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for file in processed_files:
            file_path = os.path.join(source_dir, file)
            archive.write(file_path, arcname=file)
    
    return archive_path