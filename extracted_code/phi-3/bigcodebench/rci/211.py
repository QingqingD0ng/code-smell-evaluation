import requests
import os
import zipfile
import io

def task_func(url, destination_directory, headers=None):
    if headers is None:
        headers = {'accept': 'application/octet-stream'}
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        zip_buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                zip_buffer.write(chunk)
        
        zip_path = os.path.join(destination_directory, 'downloaded.zip')
        with open(zip_path, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_directory)
        
        extracted_files = [os.path.join(destination_directory, name) for name in zip_ref.namelist()]
        return [os.path.basename(name) for name in extracted_files]
    except requests.exceptions.HTTPError as err:
        raise Exception(f"HTTP error occurred: {err.response.status_code}")
    except requests.exceptions.RequestException as err:
        raise Exception(f"Error during requests to {url}: {err}")
    except Exception as err:
        raise Exception(f"Unexpected error: {err}")