import subprocess
import os
import glob

ARCHIVE_DIR = '/tmp/archive'

def task_func(pattern):
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        
    files = glob.glob(pattern)
    if not files:
        return None
    
    archive_file = f"{ARCHIVE_DIR}/archive_{int(time.time())}.tar.gz"
    with open(archive_file, 'wb') as f:
        subprocess.run(['tar', '-czf', '-', *files], stdout=f)
    
    for file in files:
        os.remove(file)
    
    return archive_file