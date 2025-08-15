from pathlib import Path
from typing import Optional, Union
import shutil
import subprocess

def prepare_repository_from_archive(
    archive_path: str,
    filename: Optional[str] = None,
    tmp_path: Union[Path, str] = "/tmp"
) -> str:
    tmp_path = Path(tmp_path)
    if not tmp_path.exists():
        tmp_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = Path(archive_path).name
    
    extracted_path = tmp_path / filename
    
    # Uncompress the archive
    subprocess.check_call(['tar', 'xf', archive_path, '-C', str(tmp_path)])
    
    # Assuming the archive contains only one directory that we want to use as a repo
    repo_dir = next(tmp_path.rglob('*'))
    
    # Construct the repo URL
    repo_url = f"file://{repo_dir.as_posix()}"
    
    return repo_url