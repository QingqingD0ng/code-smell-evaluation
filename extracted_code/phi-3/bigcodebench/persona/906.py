import zipfile

import os

import re

import shutil


def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    processed_files = [f for f in os.listdir(source_dir) if re.search('_processed$', f)]
    target_archive_path = os.path.join(target_dir, archive_name)
    with zipfile.ZipFile(target_archive_path, 'w') as archive:
        for file in processed_files:
            archive.write(os.path.join(source_dir, file), arcname=file)
    return target_archive_path