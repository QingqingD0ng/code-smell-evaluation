import requests
from pathlib import Path
import shutil
import tarfile

def get_repo_archive(url: str, destination_path: Path) -> Path:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(destination_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with tarfile.open(destination_path, 'r:gz') as tar:
        tar.extractall(path=destination_path.parent)

    desc_file_path = next((item.name for item in tar.getmembers() if 'desc' in item.name), None)
    if desc_file_path:
        return destination_path.parent / desc_file_path
    else:
        raise FileNotFoundError("No 'desc' file found in the archive.")