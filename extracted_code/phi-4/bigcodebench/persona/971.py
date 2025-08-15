import os
from pathlib import Path
from datetime import datetime, timezone

def task_func(directory_path: str):
    if not directory_path:
        return []
    
    if not os.path.exists(directory_path):
        raise ValueError("The provided directory does not exist.")
    
    result = []
    for entry in Path(directory_path).iterdir():
        if entry.is_file():
            size = entry.stat().st_size
            try:
                creation_time = datetime.fromtimestamp(entry.stat().st_ctime, tz=timezone.utc).isoformat()
            except AttributeError:
                creation_time = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat()
            modification_time = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat()
            result.append((entry.name, size, creation_time, modification_time))
    
    return result