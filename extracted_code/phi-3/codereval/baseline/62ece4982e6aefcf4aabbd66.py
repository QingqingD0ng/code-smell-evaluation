import os

def was_processed(processed, path_name, verbose):
    absolute_pathname = os.path.abspath(path_name)
    if absolute_pathname in processed:
        if verbose:
            print(f"Processed: {absolute_pathname}")
        return True
    processed.add(absolute_pathname)
    return False