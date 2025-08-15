import subprocess
import os
import glob

ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    # Ensure the archive directory exists
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    
    # Find files matching the pattern
    files_to_archive = glob.glob(pattern)
    
    # Create an archive file name
    archive_file = os.path.join(ARCHIVE_DIR, os.path.basename(pattern) + '.tar.gz')
    
    # Archive files
    subprocess.run(['tar', '-czf', archive_file] + files_to_archive, check=True)
    
    # Delete original files
    for file_path in files_to_archive:
        os.remove(file_path)
    
    return archive_file