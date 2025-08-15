import re
import os
import shutil
from datetime import datetime

def task_func(directory):
    file_movement = {}

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                content = file.read()
            
            match = re.search(r'(?<!\[)[^\[\]]+', content)
            if match:
                subdir_name = match.group(0)
                subdir_path = os.path.join(directory, subdir_name)
                
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
                
                new_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                new_filepath = os.path.join(subdir_path, new_filename)
                
                shutil.move(filepath, new_filepath)
                
                if subdir_name not in file_movement:
                    file_movement[subdir_name] = []
                file_movement[subdir_name].append(new_filename)

    return directory, file_movement