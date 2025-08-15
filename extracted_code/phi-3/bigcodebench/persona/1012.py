import requests
from pathlib import Path
import zipfile

def task_func(url, filename):
    DOWNLOAD_DIR = Path("downloads")
    ZIP_DIR = Path("unzipped_files")

    try:
        DOWNLOAD_DIR.mkdir(exist_ok=True)
        ZIP_DIR.mkdir(exist_ok=True)

        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        zip_path = DOWNLOAD_DIR / filename
        with open(zip_path, 'wb') as file:
            file.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)

        # Get list of extracted files
        extracted_files = [file.name for file in ZIP_DIR.glob('**/*') if file.is_file()]

        return ("Download and extraction successful", extracted_files)

    except requests.exceptions.RequestException as e:
        return ("Error", [])
    except zipfile.BadZipFile:
        return ("Error", [])
    except Exception as e:
        return ("Error", [])

# Example usage:
# result = task_func('http://example.com/myfile.zip','myfile.zip')
# print(result)