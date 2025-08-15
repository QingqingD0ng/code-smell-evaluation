import zipfile
import os
import re
import shutil

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    # Create the archive
    archive_path = os.path.join(target_dir, archive_name)
    with zipfile.ZipFile(archive_path, 'w') as archive:
        # Walk through the source directory
        for root, dirs, files in os.walk(source_dir):
            # Add files that have '_processed' in their name
            for file in files:
                if '_processed' in file:
                    # Construct full file path
                    file_path = os.path.join(root, file)
                    # Add file to archive with relative path
                    archive.write(file_path, os.path.relpath(file_path, source_dir))
    return os.path.join(target_dir, archive_name)