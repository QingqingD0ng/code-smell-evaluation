import urllib.request
import os
import hashlib
import tarfile

TARGET_TAR_FILE = "downloaded_files.tar.gz"
EXPECTED_MD5_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_tar_file(tar_file):
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall()

def task_func(url):
    try:
        # Download the file
        urllib.request.urlretrieve(url, TARGET_TAR_FILE)

        # Validate MD5 checksum
        if calculate_md5(TARGET_TAR_FILE) == EXPECTED_MD5_CHECKSUM:
            # Extract contents
            extract_tar_file(TARGET_TAR_FILE)
            return True
        else:
            # Remove the file if checksum doesn't match
            os.remove(TARGET_TAR_FILE)
            return False
    except Exception as e:
        # Clean up in case of failure
        if os.path.exists(TARGET_TAR_FILE):
            os.remove(TARGET_TAR_FILE)
        return False

# Example usage
# task_func('http://example.com/files.tar.gz')