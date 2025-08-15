import re
import os
import zipfile

def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            match = re.match(pattern, filename)
            if match:
                prefix = match.group(1)
                extract_to = os.path.join(directory, prefix)
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                extracted_dirs.append(extract_to)
    return extracted_dirs