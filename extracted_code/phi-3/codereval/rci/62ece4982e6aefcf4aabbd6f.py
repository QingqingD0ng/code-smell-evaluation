import requests
from pathlib import Path
import tarfile

def get_repo_archive(url: str, destination_path: Path) -> Path:
    # Validate URL and destination path
    if not url.endswith('.tar.gz'):
        raise ValueError("URL must point to a.tar.gz archive")
    destination_path.parent.mkdir(parents=True, exist_ok=True)  # Create destination directory if not exists

    # Download the archive
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as e:
        response.close()
        raise e

    # Write the response content to a file
    with open(destination_path, 'wb') as file:
        file.write(response.content)

    # Extract the archive
    try:
        with tarfile.open(destination_path, 'r:gz') as tar:
            tar.extractall(path=destination_path.parent)
    except tarfile.ReadError as e:
        destination_path.unlink()  # Remove the corrupted archive file
        raise e

    return destination_path.parent