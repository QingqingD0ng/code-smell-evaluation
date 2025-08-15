import requests
from pathlib import Path
import tarfile

def get_repo_archive(url: str, destination_path: Path) -> Path:
    response = requests.get(url)
    response.raise_for_status()
    with open(destination_path, 'wb') as file:
        file.write(response.content)
    
    with tarfile.open(destination_path, 'r:gz') as tar:
        tar.extractall(path=destination_path.parent)
    return destination_path.parent