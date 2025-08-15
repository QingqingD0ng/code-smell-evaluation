from pathlib import Path
import shutil
from typing import Optional

def prepare_repository_from_archive(archive_path: str, filename: Optional[str] = None, tmp_path: Union[Path, str] = "/tmp") -> str:
    archive_path = Path(archive_path)
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    
    if filename:
        extract_path = tmp_path / filename
    else:
        extract_path = tmp_path / archive_path.name

    if archive_path.is_file():
        shutil.unpack_archive(str(archive_path), str(extract_path))
    else:
        raise FileNotFoundError(f"The archive {archive_path} does not exist.")

    repo_url = f"git@github.com:user/repo.git"
    return repo_url

# Example usage:
try:
    repo_url = prepare_repository_from_archive("/path/to/archive.zip", "extracted_files")
    print(repo_url)
except FileNotFoundError as e:
    print(e)