import os
from pathlib import Path
from datetime import datetime, timezone

def task_func(directory_path: str):
    if not directory_path:
        return []

    path = Path(directory_path)
    if not path.is_dir():
        raise ValueError("The provided directory does not exist.")

    files_info = []
    for file in path.iterdir():
        if file.is_file():
            file_size = file.stat().st_size
            try:
                creation_time = datetime.fromtimestamp(file.stat().st_ctime, tz=timezone.utc).isoformat()
            except AttributeError:
                creation_time = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc).isoformat()
            modification_time = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc).isoformat()
            files_info.append((file.name, file_size, creation_time, modification_time))

    return files_info