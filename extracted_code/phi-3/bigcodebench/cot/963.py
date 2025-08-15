import os
import glob
from pathlib import Path
import zipfile

def task_func(source_directory, target_directory, zip_name):
    source_directory = Path(source_directory)
    if not source_directory.is_dir():
        raise OSError(f"Source directory '{source_directory}' does not exist.")
    target_directory = Path(target_directory)
    if not target_directory.is_dir():
        target_directory.mkdir(parents=True)
    zip_path = target_directory / f"{zip_name}.zip"
    valid_extensions = ('.txt', '.docx', '.xlsx', '.csv')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in glob.glob(str(source_directory / '*' + valid_extensions)):
            zipf.write(file, os.path.relpath(file, source_directory))
    return str(zip_path)