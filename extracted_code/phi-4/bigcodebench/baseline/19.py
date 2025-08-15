import os
import glob
import zipfile

def task_func(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    files = glob.glob(os.path.join(directory, '*'))
    zip_path = os.path.join(directory, 'files.zip')
    
    if not files:
        return None
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in files:
            if os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
    
    return zip_path