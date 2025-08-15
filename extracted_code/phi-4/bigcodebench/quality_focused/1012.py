import requests
from pathlib import Path
import zipfile

DOWNLOAD_DIR = Path("downloads")
ZIP_DIR = Path("unzipped_files")

def task_func(url, filename):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)

    file_path = DOWNLOAD_DIR / filename

    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)

        # Get list of extracted files
        extracted_files = [f.name for f in ZIP_DIR.iterdir()]
        return "Download and extraction successful", extracted_files

    except (requests.RequestException, zipfile.BadZipFile, IOError) as e:
        return f"Error: {e}", []

# Example usage
# print(task_func('http://example.com/myfile.zip','myfile.zip'))