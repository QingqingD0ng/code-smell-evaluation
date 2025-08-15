import re
import os
import shutil
from datetime import datetime

DATE_FORMAT = '%Y%m%d_%H%M%S'

def task_func(directory):
    organized_files = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                match = re.search(r'\[(?:[^\]]*?)\](.*?)\s', content)
                if match:
                    subdir_name = match.group(1).strip()
                    subdir_path = os.path.join(directory, subdir_name)
                    os.makedirs(subdir_path, exist_ok=True)
                    timestamp = datetime.now().strftime(DATE_FORMAT)
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{timestamp}_{name}_{filename}"
                    new_filepath = os.path.join(subdir_path, new_filename)
                    shutil.move(filepath, new_filepath)
                    if subdir_name not in organized_files:
                        organized_files[subdir_name] = []
                    organized_files[subdir_name].append(new_filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return directory, organized_files