import re
import os
import zipfile

def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []
    if not os.path.isdir(directory):
        return extracted_dirs
    
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            file_path = os.path.join(directory, filename)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            prefix = re.match(pattern, filename).group(1)
            extract_path = os.path.join(directory, prefix)
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            try:
                zip_ref.extractall(extract_path)
                extracted_dirs.append(extract_path)
            except zipfile.BadZipFile:
                print(f"The file {filename} is not a zip file or is corrupted.")
            finally:
                zip_ref.close()
    return extracted_dirs