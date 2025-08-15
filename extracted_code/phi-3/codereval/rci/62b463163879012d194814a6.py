import zipfile

from typing import List


def files_list_from_zipfile(zip_path: str) -> List[str]:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("The provided path is not a valid zip file.")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            return zip_ref.namelist()
    except (FileNotFoundError, zipfile.BadZipFile) as e:
        raise RuntimeError("An error occurred while accessing the zip file.") from e