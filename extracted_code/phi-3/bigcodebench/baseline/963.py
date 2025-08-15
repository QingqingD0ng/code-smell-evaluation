import os
import glob
from pathlib import Path
import zipfile

def task_func(source_directory, target_directory, zip_name):
    os.makedirs(target_directory, exist_ok=True)
    
    extensions = ('.txt', '.docx', '.xlsx', '.csv')
    files_to_zip = []
    
    for ext in extensions:
        files_to_zip.extend(glob.glob(os.path.join(source_directory, f'*{ext}')))
    
    zip_path = os.path.join(target_directory, f"{zip_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))
    
    return os.path.join(target_directory, f"{zip_name}.zip")