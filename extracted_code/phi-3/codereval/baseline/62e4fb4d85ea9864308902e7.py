import os

def normalize_cmd(cmd: tuple[str,...]) -> tuple[str,...]:
    return tuple(os.path.normpath(arg) for arg in cmd)