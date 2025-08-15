import os
import glob
import zipfile

def task_func(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    zip_path = 'files.zip'
    file_paths = glob.glob(os.path.join(directory, '*'))
    
    if not file_paths:
        return None
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in file_paths:
            if os.path.isfile(file_path):
                zipf.write(file_path, os.path.relpath(file_path, directory))
    
    return zip_path

# Example usage:
# path = task_func('/path/to/files')
# print(isinstance(path, str))