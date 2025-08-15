import os

def match_file_by_prefix(prefix, file_path):
    basename = os.path.basename(file_path)
    return basename.startswith(prefix + "-") or basename.startswith(prefix + ".")