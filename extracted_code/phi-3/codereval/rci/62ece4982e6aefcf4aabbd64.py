import os

def strip_root(path, root):
    if path is None or root is None:
        raise ValueError("Path and root must not be None")
    if not path.startswith(root):
        raise ValueError("Path does not start with the root")
    return path[len(root):]