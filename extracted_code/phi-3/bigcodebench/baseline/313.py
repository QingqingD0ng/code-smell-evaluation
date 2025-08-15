import re
import os
import shutil
from datetime import datetime

def task_func(directory):
    organized_dir = directory
    files_moved = {}
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                match = re.search(r'\[(.*?)\]', content)
                if match:
                    text = match.group(1)
                else:
                    continue
                subdir = os.path.join(organized_dir, text)
                if subdir not in files_moved:
                    if not os.path.exists(subdir):
                        os.makedirs(subdir)
                    files_moved[subdir] = []
                shutil.move(os.path.join(directory, filename), os.path.join(subdir, filename))
                files_moved[subdir].append(filename)
    return organized_dir, files_moved