import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

def extract_archive(archive_path: str, extract_dir: Union[Path, str]) -> None:
    """
    Extract an archive to the specified directory.

    :param archive_path: The archive file path.
    :param extract_dir: The directory to extract the archive to.
    """
    shutil.unpack_archive(archive_path, extract_dir=Path(extract_dir))

def find_first_file(directory: Path) -> Optional[str]:
    """
    Find the first file in the directory.

    :param directory: The directory to search for the first file.
    :return: The name of the first file or None if no files are found.
    """
    files = directory.glob('*')
    if files:
        return str(next(files))
    return None

def prepare_repository_from_archive(
    archive_path: str,
    filename: Optional[str] = None,
    tmp_path: Union[Path, str] = Path("/tmp")
) -> Optional[str]:
    """
    Given an existing archive_path, uncompress it.
    Returns a file repository URL if available, otherwise None.

    :param archive_path: The archive file path.
    :param filename: Optional file name to extract from the archive.
    :param tmp_path: The directory to create a temporary directory for extraction.
    :return: The repository URL or None if no files are found.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory(dir=tmp_path) as temp_dir:
        extract_archive(archive_path, extract_dir=Path(temp_dir))
        
        # Find the file if not provided
        if not filename:
            filename = find_first_file(Path(temp_dir))
        
        # Construct the repository URL if a file is found
        if filename:
            repo_url = f"git+file://{temp_dir.as_posix()}/{filename}"
            return repo_url
        
    return None