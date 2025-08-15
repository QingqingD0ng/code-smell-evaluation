import urllib.request
import os
import hashlib
import tarfile

TARGET_TAR_FILE = "downloaded_files.tar.gz"
EXPECTED_MD5_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"


def task_func(url):
    try:
        # Download the file
        urllib.request.urlretrieve(url, TARGET_TAR_FILE)

        # Calculate the MD5 checksum of the downloaded file
        with open(TARGET_TAR_FILE, 'rb') as f:
            file_content = f.read()
            md5_checksum = hashlib.md5(file_content).hexdigest()

        # Validate the checksum
        if md5_checksum!= EXPECTED_MD5_CHECKSUM:
            os.remove(TARGET_TAR_FILE)
            return False

        # Extract the tar.gz file
        with tarfile.open(TARGET_TAR_FILE, 'r:gz') as tar:
            tar.extractall()

        # Clean up the tar.gz file
        os.remove(TARGET_TAR_FILE)
        return True

    except Exception:
        if os.path.exists(TARGET_TAR_FILE):
            os.remove(TARGET_TAR_FILE)
        return False