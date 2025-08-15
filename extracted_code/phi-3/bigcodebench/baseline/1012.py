import requests
from pathlib import Path
import zipfile

def task_func(url, filename):
    download_dir = Path("downloads")
    zip_dir = Path("unzipped_files")

    try:
        response = requests.get(url)
        response.raise_for_status()

        file_path = download_dir / filename
        with open(file_path, 'wb') as file:
            file.write(response.content)

        zip_path = zip_dir / filename
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)

        return "Download and extraction successful", [f.name for f in zip_dir.glob('**/*') if f.is_file()]

    except requests.exceptions.RequestException:
        return "Error", []
    except (zipfile.BadZipFile, zipfile.LargeZipFile):
        return "Error", []
    except FileNotFoundError:
        return "Error", []
    except Exception as e:
        return f"Error: {e}", []