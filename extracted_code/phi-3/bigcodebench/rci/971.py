import os
from pathlib import Path
from datetime import datetime

def list_files(directory_path: str) -> list:
    return [
        (file.name, file.stat().st_size, file.stat().st_ctime.isoformat() + 'Z', file.stat().st_mtime.isoformat() + 'Z')
        for file in Path(directory_path).iterdir() if file.is_file()
    ]