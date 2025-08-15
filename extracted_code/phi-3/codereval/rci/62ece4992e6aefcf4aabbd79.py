import os
import fnmatch

def make_find_paths(find_paths):
    transformed_paths = []
    for path in find_paths:
        parts = path.split(os.sep)
        dir_parts = [part for part in parts if os.path.isdir(os.path.join(os.path.dirname(path), part))]
        glob_parts = [fnmatch.translate(part) if part not in dir_parts else '**' + part + '**' for part in parts]
        try:
            transformed_path = os.path.join(*glob_parts).replace('\\\\', '\\') + '*'
            transformed_paths.append(transformed_path)
        except Exception as e:
            print(f"Error processing path {path}: {e}")
            continue
    return tuple(transformed_paths)