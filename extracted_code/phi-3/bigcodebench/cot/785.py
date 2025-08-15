import subprocess
import os
import glob

ARCHIVE_DIR = '/tmp/archive'

def create_archive(pattern):
    archive_name = 'archive.tar.gz'
    archive_path = os.path.join(ARCHIVE_DIR, archive_name)
    subprocess.run(['tar', '-czf', archive_path, '-C', os.path.dirname(pattern), '--wildcards', pattern])
    return archive_path

def cleanup(pattern, archive_path):
    for file_path in glob.glob(pattern):
        os.remove(file_path)
    os.remove(archive_path)
    cleanup(archive_path, ARCHIVE_DIR)  # Recursively delete the archive

def task_func(pattern):
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    archive_path = create_archive(pattern)
    cleanup(pattern, archive_path)
    return archive_path