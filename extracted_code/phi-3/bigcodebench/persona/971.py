import os
from pathlib import Path
from datetime import datetime, timezone


def task_func(directory_path: str):
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError("Provided directory does not exist")

    files = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            size = file_path.stat().st_size
            creation_time = datetime.fromtimestamp(file_path.stat().st_birthtime, tz=timezone.utc).isoformat()
            modification_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
            files.append((file_path.name, size, creation_time, modification_time))
    return files