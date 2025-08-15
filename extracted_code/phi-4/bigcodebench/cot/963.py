import os
import glob
from pathlib import Path
import zipfile

def task_func(source_directory, target_directory, zip_name):
    valid_extensions = ['.txt', '.docx', '.xlsx', '.csv']
    zip_name += '.zip'
    
    if not os.path.exists(source_directory):
        raise OSError(f"Source directory {source_directory} does not exist.")
    
    os.makedirs(target_directory, exist_ok=True)
    
    zip_path = os.path.join(target_directory, zip_name)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for extension in valid_extensions:
            for file_path in glob.glob(os.path.join(source_directory, f'**/*{extension}'), recursive=True):
                zipf.write(file_path, os.path.relpath(file_path, source_directory))
    
    return zip_path