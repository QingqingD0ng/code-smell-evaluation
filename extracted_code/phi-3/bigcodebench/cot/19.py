import os
import glob
import zipfile

def task_func(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    zip_path = os.path.join(directory, 'files.zip')
    all_files = glob.glob(os.path.join(directory, '*'))
    
    if not all_files:
        return None
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in all_files:
            if os.path.isfile(file):
                zipf.write(file, os.path.relpath(file, directory))
    
    return zip_path