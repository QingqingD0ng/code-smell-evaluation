import os
from pathlib import Path
from datetime import datetime

def task_func(directory_path: str) -> list:
    if not directory_path:
        return []
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError("The directory does not exist")
    file_info_list = []
    for file in directory.iterdir():
        if file.is_file():
            stat = file.stat()
            file_info = (
                file.name,
                stat.st_size,
                datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat() + 'Z',
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() + 'Z'
            )
            file_info_list.append(file_info)
    return file_info_list