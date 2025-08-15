import requests

import os

import zipfile


def task_func(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}
    
    local_filename = url.split('/')[-1]
    local_zip_path = os.path.join(destination_directory, local_filename)
    
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(local_zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_directory)

    return [name for name in os.listdir(destination_directory) if os.path.isfile(os.path.join(destination_directory, name))]