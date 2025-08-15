import os

def strip_root(path, root):
    if path.startswith(root):
        return path[len(root):]
    else:
        raise ValueError("Path does not start with the root")