from pathlib import Path

def find_path_type(path):
    path_obj = Path(path)
    if path_obj.is_dir():
        return "Directory"
    elif path_obj.is_file():
        return "File"
    else:
        return "Unknown"