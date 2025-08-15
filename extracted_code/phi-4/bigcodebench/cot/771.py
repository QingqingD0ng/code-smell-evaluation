import re
import os
from pathlib import Path
import csv

def task_func(directory: str, pattern: str = r'^(.*?)-\d+\.csv$') -> list:
    new_files = []
    path = Path(directory)
    
    for file_path in path.glob('*.csv'):
        file_name = file_path.name
        match = re.match(pattern, file_name)
        
        if match:
            new_name = f"{match.group(1)}.csv"
            new_file_path = file_path.with_name(new_name)
            
            with open(file_path, 'r', newline='') as src_file:
                reader = csv.reader(src_file)
                with open(new_file_path, 'w', newline='') as dest_file:
                    writer = csv.writer(dest_file)
                    for row in reader:
                        writer.writerow(row)
            
            new_files.append(new_file_path.name)
    
    return new_files