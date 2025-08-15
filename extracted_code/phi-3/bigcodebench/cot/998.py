import urllib.request

import hashlib

import tarfile

import os


TARGET_TAR_FILE = "downloaded_files.tar.gz"

EXPECTED_MD5_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"


def task_func(url):
    try:
        urllib.request.urlretrieve(url, TARGET_TAR_FILE)
        with open(TARGET_TAR_FILE, "rb") as file:
            file_hash = hashlib.md5()
            while chunk := file.read(8192):
                file_hash.update(chunk)
        if file_hash.hexdigest() == EXPECTED_MD5_CHECKSUM:
            with tarfile.open(TARGET_TAR_FILE, "r:gz") as tar:
                tar.extractall()
            return True
        else:
            os.remove(TARGET_TAR_FILE)
            return False
    except Exception as e:
        if os.path.exists(TARGET_TAR_FILE):
            os.remove(TARGET_TAR_FILE)
        return False


# Example usage

# task_func('http://example.com/files.tar.gz')