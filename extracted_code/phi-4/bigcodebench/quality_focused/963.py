import os
import glob
from pathlib import Path
import zipfile

def task_func(source_directory, target_directory, zip_name):
    valid_extensions = ['.txt', '.docx', '.xlsx', '.csv']
    if not os.path.exists(source_directory):
        raise OSError(f"Source directory {source_directory} does not exist.")
    
    Path(target_directory).mkdir(parents=True, exist_ok=True)
    zip_path = Path(target_directory) / f"{zip_name}.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for ext in valid_extensions:
            for file in glob.glob(os.path.join(source_directory, f"*{ext}")):
                zipf.write(file, os.path.relpath(file, source_directory))
    
    return str(zip_path)