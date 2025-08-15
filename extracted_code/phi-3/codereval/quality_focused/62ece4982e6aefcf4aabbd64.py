import os

def strip_root(path, root):
    if path.startswith(root):
        return path[len(root):]
    else:
        raise ValueError(f"The path does not start with the root '{root}'")

# Example usage:
try:
    full_path = "/home/user/docs/report.txt"
    root_path = "/home/user/docs"
    relative_path = strip_root(full_path, root_path)
    print(relative_path)  # Output: "report.txt"
except ValueError as e:
    print(e)