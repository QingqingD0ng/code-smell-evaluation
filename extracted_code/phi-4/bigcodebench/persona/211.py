import requests
import os
import zipfile

def task_func(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Download the zip file
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    zip_filename = os.path.join(destination_directory, 'downloaded.zip')
    
    # Write the zip file to disk
    with open(zip_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Extract the zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        extracted_files = zip_ref.namelist()
        zip_ref.extractall(destination_directory)

    # Clean up the downloaded zip file
    os.remove(zip_filename)

    return extracted_files