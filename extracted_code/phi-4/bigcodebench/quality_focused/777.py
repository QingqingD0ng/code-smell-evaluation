import re
import os
import zipfile

def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []
    for file_name in os.listdir(directory):
        if re.match(pattern, file_name):
            prefix_match = re.match(pattern, file_name)
            if prefix_match:
                prefix = prefix_match.group(1)
                extract_path = os.path.join(directory, prefix)
                zip_path = os.path.join(directory, file_name)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                extracted_dirs.append(extract_path)
    return extracted_dirs