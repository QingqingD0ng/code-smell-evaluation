import os
from pathlib import Path
from datetime import datetime

def task_func(directory_path: str):
    if not directory_path:
        return []
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError("Provided path does not exist or is not a directory")
    
    file_info_list = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            file_info = (
                file_path.name,
                file_path.stat().st_size,
                datetime.fromtimestamp(file_path.stat().st_ctime, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
            )
            file_info_list.append(file_info)
    return file_info_list