import requests
import os
import zipfile

def download_and_extract_zip(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    zip_path = os.path.join(destination_directory, 'downloaded.zip')
    with open(zip_path, 'wb') as file:
        file.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_directory)

    extracted_files = [os.path.join(destination_directory, name) for name in zip_ref.namelist()]
    os.remove(zip_path)

    return extracted_files