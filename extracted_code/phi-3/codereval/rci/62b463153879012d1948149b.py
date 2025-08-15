import os
from typing import Optional

def match_file_by_prefix(prefix: str, file_path: str) -> Optional[bool]:
    try:
        basename = os.path.basename(file_path)
        return basename.startswith(prefix + "-") or basename.startswith(prefix + ".")
    except OSError:
        return None