import requests
import os
import zipfile

def task_func(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

    zip_filename = os.path.join(destination_directory, os.path.basename(url))
    with open(zip_filename, 'wb') as zip_file:
        zip_file.write(response.content)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(destination_directory)

    extracted_files = [os.path.basename(name) for name in zip_ref.namelist()]
    return extracted_files

# Example usage:
# extracted_files = task_func("https://example.com/data.zip", "/path/to/destination")
# print(extracted_files)