import os
import requests
from zipfile import ZipFile, BadZipFile

def download_and_extract(url, download_path="mnt/data/downloads/"):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not validate_url(url):
        return "Error: Invalid URL. Please provide a valid and reachable URL."

    file_name = url.split('/')[-1]
    file_path = os.path.join(download_path, file_name)

    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, 'wb') as file:
            file.write(response.content)

        if not validate_file_mime_type(file_path):
            return "Error: The downloaded file is not a valid ZIP file."

        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)

        return os.path.join(download_path, file_name.replace('.zip', ''))
    except requests.exceptions.HTTPError:
        return "Error: Unable to download the file from the provided URL."
    except BadZipFile:
        return "Error: The downloaded file is not a valid ZIP file."
    except Exception as e:
        return f"Error: {str(e)}"

def validate_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False

def validate_file_mime_type(file_path):
    mime_type = requests.head(file_path, headers={'Accept': 'application/json'}).headers.get('Content-Type', '')
    return mime_type == 'application/zip'