from pathlib import Path

def was_processed(processed, path_name, verbose):
    path = Path(path_name).resolve()
    if path in processed:
        if verbose:
            print(f"Entry already processed: {path}")
        return True
    processed.add(path)
    return False