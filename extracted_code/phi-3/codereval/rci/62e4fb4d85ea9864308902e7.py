import os
from pathlib import Path
from typing import Tuple

def normalize_cmd(cmd: Tuple[str,...]) -> Tuple[str,...]:
    normalized_cmd = []

    for arg in cmd:
        # Check if the argument is a valid file path
        if Path(arg).is_file():
            normalized_arg = os.path.normpath(arg)
            # Check if it's a valid executable
            if os.access(normalized_arg, os.X_OK):
                normalized_arg += ".exe"
            normalized_cmd.append(normalized_arg)
        else:
            raise ValueError(f"Invalid file path: {arg}")

    return tuple(normalized_cmd)

# Test the function with a sample command
sample_cmd = ("my_app", "input.txt", "output.txt")
print(normalize_cmd(sample_cmd))