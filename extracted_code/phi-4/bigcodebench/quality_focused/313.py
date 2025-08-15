import re
import os
import shutil
from datetime import datetime


def task_func(directory):
    organized_files = {}
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        with open(file_path, 'r') as file:
            content = file.read()
            match = re.search(r'(?<!\[)[^\[\]]+", content)
            
            if match:
                folder_name = match.group(0).strip()
                folder_path = os.path.join(directory, folder_name)
                
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                new_filename = f"{filename.split('.')[0]}_{timestamp}.{filename.split('.')[1]}"
                new_file_path = os.path.join(folder_path, new_filename)
                
                shutil.move(file_path, new_file_path)
                
                if folder_name not in organized_files:
                    organized_files[folder_name] = []
                organized_files[folder_name].append(new_filename)
    
    return directory, organized_files