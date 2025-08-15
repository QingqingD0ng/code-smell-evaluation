import os
from pathlib import Path
import shutil
import sys

def task_func(kwargs, target_dir="non_none_files"):
    target_path = Path(target_dir)
    copied_files = []
    success_count = 0
    failed_count = 0

    for file_path, content in kwargs.items():
        if content is not None and isinstance(content, str):
            path = Path(file_path)

            if path.exists() and path.stat().st_size > 0:
                target_file_path = target_path / path.name

                try:
                    shutil.copy(file_path, target_file_path)
                    copied_files.append(str(target_file_path))
                    success_count += 1
                except Exception as e:
                    print(f"Error copying {file_path}: {e}", file=sys.stderr)
                    failed_count += 1

    print(f"Successfully copied {success_count} out of {len(kwargs)} files.", file=sys.stderr)
    if failed_count > 0:
        print(f"Failed to copy {failed_count} files.", file=sys.stderr)

    return copied_files

# Example usage
files = {
    '/path/to/file1.txt': 'Hello',
    '/path/to/file2.txt': None,
    '/path/to/file3.txt': 'World',
    '/path/to/file4.txt': 'Another',
    '/path/to/file5.txt': 'Example'
}

copied_files = task_func(files)
print(copied_files)