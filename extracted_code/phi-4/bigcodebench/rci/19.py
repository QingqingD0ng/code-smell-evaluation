import os
import glob
import zipfile

ZIP_FILENAME = 'files.zip'

def zip_files_in_directory(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    file_pattern = os.path.join(directory, '*')
    files = glob.glob(file_pattern)
    zip_path = os.path.join(directory, ZIP_FILENAME)
    
    if not files:
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files:
                if os.path.isfile(file):
                    zipf.write(file, os.path.basename(file))
    except Exception as e:
        print(f"An error occurred while creating the zip file: {e}")
        return None
    
    return zip_path