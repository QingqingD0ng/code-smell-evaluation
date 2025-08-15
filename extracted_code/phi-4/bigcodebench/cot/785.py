import subprocess
import os
import glob
import shutil
from datetime import datetime

ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    
    matched_files = glob.glob(pattern)
    if not matched_files:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file = os.path.join(ARCHIVE_DIR, f"archive_{timestamp}.tar.gz")
    
    with shutil.make_archive(archive_file.replace('.tar.gz', ''), 'gztar', '.', pattern) as archive:
        pass
    
    for file in matched_files:
        os.remove(file)
    
    return archive_file

# Example usage:
# archive_path = task_func('*.txt')