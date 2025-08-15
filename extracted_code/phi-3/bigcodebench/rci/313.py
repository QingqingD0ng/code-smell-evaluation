import re
import os
import shutil
from datetime import datetime

def parse_filename_for_subdirectory(filename):
    """Extract text from filename for subdirectory name."""
    basename = os.path.basename(filename)
    match = re.search(r'(\w+)(\.[^.]+)$', basename)
    return match.group(1) if match else None

def create_subdirectory(directory, subdir_name):
    """Create a subdirectory if it doesn't exist."""
    subdir_path = os.path.join(directory, subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    return subdir_path

def move_file(source, destination):
    """Move a file from source to destination."""
    shutil.move(source, destination)

def task_func(directory):
    organized_dir = directory
    files_moved = {}
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            subdir_name = parse_filename_for_subdirectory(filename)
            if subdir_name:
                subdir = create_subdirectory(organized_dir, subdir_name)
                move_file(os.path.join(organized_dir, filename), subdir)
                files_moved.setdefault(subdir, []).append(filename)
    return organized_dir, files_moved

# Example usage:
temp_dir = "path_to_temporary_directory"  # Replace with your actual temporary directory path
dir_after_organization, files_moved = task_func(temp_dir)
print(f"Directory after organization: {dir_after_organization}")
print("Files moved to each subdirectory:")
for subdir, files in files_moved.items():
    print(f"{subdir}: {files}")