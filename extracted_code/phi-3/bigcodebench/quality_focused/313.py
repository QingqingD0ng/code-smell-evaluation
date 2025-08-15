import re
import os
import shutil
from datetime import datetime

def task_func(directory):
    organized_dir = directory
    files_moved = {}

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read(1000)  # Read the first 1000 characters
                match = re.search(r'\[(.*?)\]', content)
                if match:
                    extracted_text = match.group(1)
                else:
                    continue  # Skip files with no extractable text

                subdir = os.path.join(organized_dir, extracted_text)
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                
                new_filename = re.sub(r'\[.*?\]', '', content).strip() + '_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
                shutil.move(file_path, os.path.join(subdir, new_filename))
                files_moved.setdefault(subdir, []).append(new_filename)

    return organized_dir, files_moved