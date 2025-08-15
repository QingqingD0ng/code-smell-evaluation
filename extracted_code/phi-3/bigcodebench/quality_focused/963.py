import os
import glob
from pathlib import Path
import zipfile

def task_func(source_directory, target_directory, zip_name):
    source_directory = Path(source_directory)
    target_directory = Path(target_directory)
    zip_name_with_extension = zip_name + '.zip'
    zip_file_path = target_directory / zip_name_with_extension

    if not source_directory.exists():
        raise OSError(f"Source directory {source_directory} does not exist.")

    target_directory.mkdir(parents=True, exist_ok=True)

    valid_extensions = ('.txt', '.docx', '.xlsx', '.csv')
    files_to_zip = [file for file in source_directory.glob('**/*') if file.suffix in valid_extensions]

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file, arcname=file.name)

    return str(zip_file_path)