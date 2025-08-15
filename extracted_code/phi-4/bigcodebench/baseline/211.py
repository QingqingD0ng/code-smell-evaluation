import requests
import os
import zipfile
from io import BytesIO

def task_func(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}
        
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(destination_directory)
        return z.namelist()