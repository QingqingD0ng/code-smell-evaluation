import urllib.request
import os
import hashlib
import tarfile

TARGET_TAR_FILE = "downloaded_files.tar.gz"
EXPECTED_MD5_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"

def task_func(url):
    try:
        response = urllib.request.urlopen(url)
        file_data = response.read()
        file_md5 = hashlib.md5(file_data).hexdigest()
        if file_md5 == EXPECTED_MD5_CHECKSUM:
            with open(TARGET_TAR_FILE, "wb") as file:
                file.write(file_data)
            tar = tarfile.open(TARGET_TAR_FILE, "r:gz")
            tar.extractall()
            tar.close()
            os.remove(TARGET_TAR_FILE)
            return True
        else:
            os.remove(TARGET_TAR_FILE)
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False