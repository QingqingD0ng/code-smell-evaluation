import requests
from pathlib import Path
import zipfile

DOWNLOAD_DIR = Path("downloads")
ZIP_DIR = Path("unzipped_files")

def task_func(url, filename):
    try:
        DOWNLOAD_DIR.mkdir(exist_ok=True)
        ZIP_DIR.mkdir(exist_ok=True)

        response = requests.get(url)
        response.raise_for_status()

        file_path = DOWNLOAD_DIR / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)

        unzipped_files = list(ZIP_DIR.iterdir())
        filenames = [file.name for file in unzipped_files]

        return "Download and extraction successful", filenames

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}", []
    except (FileNotFoundError, zipfile.BadZipFile) as e:
        return f"Error: {str(e)}", []

# Example usage
# print(task_func('http://example.com/myfile.zip','myfile.zip'))