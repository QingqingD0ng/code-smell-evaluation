from requests import get, RequestException
from pathlib import Path
import zipfile
import tempfile

DOWNLOAD_DIR = Path("downloads")
ZIP_DIR = Path("unzipped_files")

def task_func(url, filename):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DOWNLOAD_DIR / filename
    try:
        response = get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)
        return "Download and extraction successful", list(ZIP_DIR.iterdir())
    except RequestException as e:
        return f"Error: {e}"
    except zipfile.BadZipFile:
        return "Error: The downloaded file is not a valid zip file."
    except FileNotFoundError:
        return "Error: The downloaded file could not be found."
    except Exception as e:
        return f"Error: {e}"