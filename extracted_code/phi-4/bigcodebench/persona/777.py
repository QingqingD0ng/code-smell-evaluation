import re
import os
import zipfile

def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            full_path = os.path.join(directory, filename)
            match = re.match(pattern, filename)
            if match:
                extract_dir = os.path.join(directory, match.group(1))
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(full_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted_dirs.append(extract_dir)
    return extracted_dirs, []

# Example usage:
# print(task_func('/tmp/my_data'))