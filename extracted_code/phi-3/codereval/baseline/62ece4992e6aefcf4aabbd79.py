import fnmatch

def make_find_paths(find_paths):
    transformed_paths = []
    for path in find_paths:
        transformed_path = []
        for part in path.split(os.sep):
            if os.path.isdir(os.path.join(os.path.dirname(path), part)):
                part = '**' + part + '**'
            transformed_path.append(fnmatch.translate(part))
        transformed_paths.append('*' + os.path.join(*transformed_path).replace('\\\\', '\\') + '*')
    return tuple(transformed_paths)