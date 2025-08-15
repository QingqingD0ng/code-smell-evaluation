import re
import os
import shutil
from datetime import datetime

def task_func(directory):
    organized_files = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                content = file.read()
            match = re.search(r'\[(?:[^\]]*?)\](.*?)\s', content)
            if match:
                subdir_name = match.group(1).strip()
                subdir_path = os.path.join(directory, subdir_name)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
                new_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                new_filepath = os.path.join(subdir_path, new_filename)
                shutil.move(filepath, new_filepath)
                if subdir_name not in organized_files:
                    organized_files[subdir_name] = []
                organized_files[subdir_name].append(new_filename)
    return directory, organized_files