import requests
from pathlib import Path
import zipfile

# Constants
DOWNLOAD_DIR = Path("downloads")
ZIP_DIR = Path("unzipped_files")

def task_func(url, filename):
    download_path = DOWNLOAD_DIR / filename
    unzip_dir = ZIP_DIR / Path(filename).stem

    # Ensure directories exist
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)

    try:
        # Download the zip file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

        # List files in the unzipped directory
        extracted_files = [f.name for f in unzip_dir.iterdir() if f.is_file()]

        return ("Download and extraction successful", extracted_files)

    except requests.exceptions.RequestException as e:
        return (f"Error: Network-related issue - {e}", [])
    except (FileNotFoundError, zipfile.BadZipFile) as e:
        return (f"Error: File-related issue - {e}", [])
    except Exception as e:
        return (f"Error: Unexpected issue - {e}", [])