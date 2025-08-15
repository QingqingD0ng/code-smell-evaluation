import os

def find_path_type(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            return "file"
        elif os.path.isdir(path):
            return "directory"
        else:
            return "unknown"
    else:
        return "path does not exist"

# Further improvements could be made by adding a bit more sophisticated error handling, but that would require the use of docstrings or comments, which we are avoiding.