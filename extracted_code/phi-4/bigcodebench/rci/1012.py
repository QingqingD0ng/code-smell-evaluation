import requests
from pathlib import Path
import zipfile
import os

def task_func(url, filename, download_dir=Path("downloads"), zip_dir=Path("unzipped_files")):
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = download_dir / filename
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, 'wb') as file:
            file.write(response.content)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)
        
        extracted_files = os.listdir(zip_dir)
        
        return ("Download and extraction successful", extracted_files)
    
    except requests.exceptions.RequestException:
        return ("Error: Network-related exception occurred", [])
    
    except (FileNotFoundError, PermissionError):
        return ("Error: File-related exception occurred", [])
    
    except zipfile.BadZipFile:
        return ("Error: Bad zip file", [])
    
    finally:
        if file_path.exists():
            file_path.unlink()

        # Clean up extracted files if needed
        for item in zip_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for sub_item in item.iterdir():
                    sub_item.unlink()
                item.rmdir()