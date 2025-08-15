import shutil
from typing import Tuple

def normalize_cmd(cmd: Tuple[str,...]) -> Tuple[str,...]:
    return tuple(shutil.which(c) or c for c in cmd)