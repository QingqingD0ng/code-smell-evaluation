import subprocess
import os
import glob

ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    archive_file = os.path.join(ARCHIVE_DIR, os.path.basename(glob.glob(pattern)[0]) + '.tar.gz')
    subprocess.run(['tar', '-czf', archive_file, '--null', '--files-from=-', pattern])
    for file in glob.glob(pattern):
        os.remove(file)
    return archive_file