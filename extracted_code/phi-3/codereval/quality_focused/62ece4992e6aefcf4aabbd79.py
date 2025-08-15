def make_find_paths(find_paths):
    glob_patterns = []
    for path in find_paths:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        parts = path.split(os.sep)
        glob_pattern = os.path.join(os.sep.join(parts[:-1]), '*' + parts[-1])
        glob_patterns.append(glob_pattern)
    return tuple(glob_patterns)