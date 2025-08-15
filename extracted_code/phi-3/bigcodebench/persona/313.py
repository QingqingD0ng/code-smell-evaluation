import re
import os
import shutil
from datetime import datetime


def task_func(directory):
    pattern = re.compile(r'\[(.*?)\]')
    subdirs = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            match = pattern.search(filename)
            if match:
                subdir_name = match.group(1)
                subdir_path = os.path.join(directory, subdir_name)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
                shutil.move(filepath, os.path.join(subdir_path, filename))
                subdirs.setdefault(subdir_name, []).append(filename)
    return directory, subdirs


# Example usage:
# temp_dir = tempfile.mkdtemp()
# create_test_directory(temp_dir, {"file1.txt": "subdir1[content]", "file2.txt": "subdir1[content]", "file3.txt": "subdir2[content]"})
# dir, files = task_func(temp_dir)
# print(files['subdir2'][0].startswith('file3_'))  # Should print True

# Cleanup:
# os.removedirs(temp_dir)