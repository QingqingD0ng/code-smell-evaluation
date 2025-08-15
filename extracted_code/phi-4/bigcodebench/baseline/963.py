import os
import glob
from pathlib import Path
import zipfile

def task_func(source_directory, target_directory, zip_name):
    if not os.path.exists(source_directory):
        raise OSError(f"Source directory {source_directory} does not exist.")
    
    extensions = ['.txt', '.docx', '.xlsx', '.csv']
    files_to_zip = []
    for ext in extensions:
        files_to_zip.extend(glob.glob(os.path.join(source_directory, f"*{ext}")))

    target_directory_path = Path(target_directory)
    target_directory_path.mkdir(parents=True, exist_ok=True)

    zip_file_path = target_directory_path / f"{zip_name}.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.relpath(file, start=source_directory))

    return str(zip_file_path)