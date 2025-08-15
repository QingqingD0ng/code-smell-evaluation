import subprocess
import os
import glob
import shutil
from tempfile import NamedTemporaryFile

# Constants
ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    # Ensure the archive directory exists
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # Find files matching the pattern
    files_to_archive = glob.glob(pattern)

    if not files_to_archive:
        raise FileNotFoundError(f"No files found matching the pattern: {pattern}")

    # Create a temporary archive file
    with NamedTemporaryFile(dir=ARCHIVE_DIR, delete=False, suffix='.tar.gz') as archive_file:
        archive_path = archive_file.name

    # Archive the files
    try:
        subprocess.check_call(['tar', '-czf', archive_path] + files_to_archive)
    except subprocess.CalledProcessError as e:
        # Clean up if archiving fails
        os.remove(archive_path)
        raise RuntimeError(f"Failed to create archive: {e}")

    # Delete the original files
    for file_path in files_to_archive:
        try:
            os.remove(file_path)
        except OSError as e:
            raise RuntimeError(f"Failed to delete file {file_path}: {e}")

    return archive_path