import subprocess
from pathlib import Path
from typing import Optional, Union
import requests

def prepare_repository_from_archive(
    archive_path: str,
    filename: Optional[str] = None,
    tmp_path: Union[Path, str] = "/tmp"
) -> str:
    tmp_path = Path(tmp_path)
    tmp_dir = tmp_path / "repo_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = Path(archive_path).name
    
    destination_path = tmp_dir / filename
    try:
        subprocess.run(["tar", "-xzf", archive_path, "-C", str(tmp_dir)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract archive: {e}")
    
    repo_url = f"https://github.com/user/repo_name.git"
    with open(destination_path / ".gitignore", "w") as gitignore:
        gitignore.write("# This.gitignore file is auto-generated\n")

    # Move the repository to the desired location or commit it directly if needed
    if tmp_dir.name!= "repo_tmp":
        (tmp_dir / destination_path.name).rename(destination_path)

    return repo_url

# Example usage:
# repo_url = prepare_repository_from_archive("/path/to/archive.tar.gz")