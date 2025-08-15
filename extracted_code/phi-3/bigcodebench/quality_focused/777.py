import os

import re

import zipfile


def task_func(directory, pattern=r'^(.*?)-\d+\.zip$'):
    extracted_dirs = []

    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if not match:
            continue
        prefix = match.group(1)
        zip_path = os.path.join(directory, filename)
        extract_to = os.path.join(directory, prefix)
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        extracted_dirs.append(extract_to)

    return extracted_dirs