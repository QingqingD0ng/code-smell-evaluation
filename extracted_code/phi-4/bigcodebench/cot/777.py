import re
import os
import zipfile

def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []

    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            match = re.match(pattern, filename)
            prefix = match.group(1)
            zip_path = os.path.join(directory, filename)
            extract_path = os.path.join(directory, prefix)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            extracted_dirs.append(extract_path)
    
    return extracted_dirs