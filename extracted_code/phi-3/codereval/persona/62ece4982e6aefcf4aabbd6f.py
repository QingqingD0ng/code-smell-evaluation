import requests
from pathlib import Path
import tarfile
import re

def get_repo_archive(url: str, destination_path: Path) -> Path:
    response = requests.get(url)
    response.raise_for_status()
    archive_path = destination_path / Path(url).name
    with open(archive_path, 'wb') as file:
        file.write(response.content)
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=destination_path)
    
    desc_files = tar.getnames()
    desc_files = [f for f in desc_files if re.search(r'\.desc$', f)]
    for desc_file in desc_files:
        desc_path = destination_path / desc_file
        with open(desc_path, 'r') as file:
            print(file.read())
    
    return destination_path