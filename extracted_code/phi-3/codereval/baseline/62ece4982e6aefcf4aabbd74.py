import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

def prepare_repository_from_archive(
    archive_path: str,
    filename: Optional[str] = None,
    tmp_path: Union[Path, str] = Path("/tmp")
) -> str:
    # Create a temporary directory
    with tempfile.TemporaryDirectory(dir=tmp_path) as temp_dir:
        # Extract the archive
        shutil.unpack_archive(archive_path, extract_dir=temp_dir)
        
        # Find the file if not provided
        if not filename:
            files = os.listdir(temp_dir)
            if not files:
                raise FileNotFoundError("No files found in the archive.")
            filename = files[0]
        
        # Construct the repository URL
        repo_url = f"git+file://{temp_dir.as_posix()}/{filename}"
        
    return repo_url