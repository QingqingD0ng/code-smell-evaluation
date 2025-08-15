def make_find_paths(find_paths):
    def to_glob_pattern(path):
        return path.replace("*", ".*").replace("?", ".")

    return tuple(to_glob_pattern(path) for path in find_paths)