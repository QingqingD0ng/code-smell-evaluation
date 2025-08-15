import os
from pathlib import Path
from datetime import datetime, timezone

def task_func(directory_path: str):
    if not directory_path:
        return []
    
    path = Path(directory_path)
    if not path.is_dir():
        raise ValueError("Provided directory does not exist.")
    
    result = []
    for file in path.iterdir():
        if file.is_file():
            file_size = file.stat().st_size
            creation_time = datetime.fromtimestamp(file.stat().st_ctime, timezone.utc).isoformat()
            modification_time = datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat()
            result.append((file.name, file_size, creation_time, modification_time))
    
    return result