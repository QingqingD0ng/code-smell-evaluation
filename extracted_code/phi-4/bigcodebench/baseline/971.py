import os
from pathlib import Path
from datetime import datetime, timezone

def task_func(directory_path: str):
    if not directory_path:
        return []
    
    directory = Path(directory_path)
    
    if not directory.is_dir():
        raise ValueError("The provided directory does not exist.")
    
    result = []
    
    for file in directory.iterdir():
        if file.is_file():
            file_size = file.stat().st_size
            creation_time = file.stat().st_ctime
            modification_time = file.stat().st_mtime
            
            creation_time_utc = datetime.fromtimestamp(creation_time, tz=timezone.utc).isoformat()
            modification_time_utc = datetime.fromtimestamp(modification_time, tz=timezone.utc).isoformat()
            
            result.append((file.name, file_size, creation_time_utc, modification_time_utc))
    
    return result