import requests
import os
import zipfile
from io import BytesIO
from zipfile import BadZipFile

def download_and_extract_zip(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}
    
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(destination_directory)
            return z.namelist()
    except requests.RequestException:
        print("HTTP error occurred during download.")
    except BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
    except Exception:
        print("An unexpected error occurred.")
    
    return []