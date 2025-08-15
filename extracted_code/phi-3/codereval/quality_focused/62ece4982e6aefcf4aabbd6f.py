import requests
import tarfile
from pathlib import Path
from urllib.parse import urlparse

def get_repo_archive(url: str, destination_path: Path):
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path.split('/')[-1])
    destination_path.mkdir(parents=True, exist_ok=True)
    destination = destination_path / filename

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    with tarfile.open(destination, "r:gz") as tar:
        tar.extractall(path=destination_path.parent)

    desc_files = list(destination_path.glob('desc*'))
    for desc_file in desc_files:
        with open(desc_file, 'r') as f:
            print(f.read())

    return destination_path