import os
import glob
from shutil import make_archive

ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    files_to_archive = glob.glob(pattern)
    archive_file = os.path.join(ARCHIVE_DIR, os.path.basename(pattern) + '.tar.gz')
    make_archive(archive_file, 'gztar', ARCHIVE_DIR, os.path.basename(pattern))
    for file_path in files_to_archive:
        os.remove(file_path)
    return archive_file