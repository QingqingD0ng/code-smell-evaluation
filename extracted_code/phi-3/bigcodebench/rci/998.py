import urllib.request
import os
import hashlib
import tarfile

TARGET_TAR_FILE = "downloaded_files.tar.gz"

def calculate_md5(file_path):
    with open(file_path, "rb") as file:
        file_data = file.read()
        return hashlib.md5(file_data).hexdigest()

def task_func(url, expected_md5_checksum):
    try:
        response = urllib.request.urlopen(url)
        with open(TARGET_TAR_FILE, "wb") as file:
            file.write(response.read())
        if calculate_md5(TARGET_TAR_FILE) == expected_md5_checksum:
            with tarfile.open(TARGET_TAR_FILE, "r:gz") as tar:
                tar.extractall()
            os.remove(TARGET_TAR_FILE)
            return True
        else:
            os.remove(TARGET_TAR_FILE)
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False