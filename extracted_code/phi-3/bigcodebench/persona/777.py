import re
import os
import zipfile

def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            prefix = re.match(pattern, filename).group(1)
            zip_path = os.path.join(directory, filename)
            extract_dir = os.path.join(directory, prefix)
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            extracted_dirs.append(extract_dir)
    return extracted_dirs