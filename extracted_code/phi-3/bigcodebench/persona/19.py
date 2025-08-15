import os
import glob
import zipfile

def task_func(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    all_files = glob.glob(os.path.join(directory, '*'))
    all_files = [file for file in all_files if os.path.isfile(file)]

    if not all_files:
        return None

    zip_path = os.path.join(directory, 'files.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in all_files:
            zipf.write(file, os.path.relpath(file, directory))

    return zip_path

os.chdir('/path/to/files')  # Change working directory to the specified directory
zip_path = task_func('/path/to/files')
print(zip_path)