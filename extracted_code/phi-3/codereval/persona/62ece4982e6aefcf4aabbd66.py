import os

def was_processed(processed, path_name, verbose=False):
    absolute_path = os.path.abspath(path_name)
    if absolute_path in processed:
        if verbose:
            print(f"Warning: '{absolute_path}' has already been processed.")
        return True
    processed.add(absolute_path)
    return False