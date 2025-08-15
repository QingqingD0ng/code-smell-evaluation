import re
import os
import shutil
from datetime import datetime

def task_func(directory):
    organized_dir = os.path.join(directory, 'organized')
    os.makedirs(organized_dir, exist_ok=True)
    move_dict = {}
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            file_path = os.path.join(directory, item)
            match = re.search(r'\[(.*?)\]', item)
            if match:
                extracted_text = match.group(1).strip()
                target_dir = os.path.join(organized_dir, extracted_text)
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(file_path, target_dir)
                if extracted_text not in move_dict:
                    move_dict[extracted_text] = []
                move_dict[extracted_text].append(item)
    return organized_dir, move_dict