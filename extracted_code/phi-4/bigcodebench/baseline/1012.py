import requests
from pathlib import Path
import zipfile
import os

DOWNLOAD_DIR = Path("downloads")
ZIP_DIR = Path("unzipped_files")

def task_func(url, filename):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)
    
    try:
        # Download zip file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save zip file
        file_path = DOWNLOAD_DIR / filename
        with open(file_path, 'wb') as file:
            file.write(response.content)
        
        # Extract zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)
        
        # List files in the unzipped directory
        extracted_files = os.listdir(ZIP_DIR)
        
        return ("Download and extraction successful", extracted_files)
    
    except requests.exceptions.RequestException as e:
        return ("Error: Network-related exception occurred", [])
    
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile) as e:
        return ("Error: File-related exception occurred", [])

    finally:
        # Clean up downloaded zip file
        if file_path.exists():
            file_path.unlink()