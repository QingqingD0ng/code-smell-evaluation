import requests
from pathlib import Path
import zipfile

DOWNLOAD_DIR = Path("downloads")
ZIP_DIR = Path("unzipped_files")

def task_func(url, filename):
    try:
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DOWNLOAD_DIR / filename
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as file:
            file.write(response.content)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)
        
        extracted_files = [name for name in ZIP_DIR.iterdir() if name.is_file()]
        return "Download and extraction successful", extracted_files
    
    except requests.exceptions.RequestException:
        return "Error", []
    except zipfile.BadZipFile:
        return "Error", []
    except Exception as e:
        return f"Error: {str(e)}", []