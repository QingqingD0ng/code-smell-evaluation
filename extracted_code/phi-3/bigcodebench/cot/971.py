import os
from pathlib import Path
from datetime import datetime, timezone

def analyze_directory(directory_path: str) -> list:
    if not directory_path:
        return []
    
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError("Provided path is not a directory")
    
    result = []
    
    for entry in directory.iterdir():
        if entry.is_file():
            size = entry.stat().st_size
            creation_time = entry.lstat().st_ctime
            modification_time = entry.stat().st_mtime
            
            result.append((entry.name, size, datetime.fromtimestamp(creation_time, timezone.utc).isoformat() + 'Z', datetime.fromtimestamp(modification_time, timezone.utc).isoformat() + 'Z'))
    
    return result