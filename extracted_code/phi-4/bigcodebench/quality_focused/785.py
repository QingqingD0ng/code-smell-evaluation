import subprocess
import os
import glob

ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    files = glob.glob(pattern)
    if not files:
        return None
    archive_file = os.path.join(ARCHIVE_DIR, 'archive.tar.gz')
    with subprocess.Popen(['tar', '-czf', archive_file] + files, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        proc.communicate()
    for file in files:
        os.remove(file)
    return archive_file