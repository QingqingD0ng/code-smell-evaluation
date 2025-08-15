import os
from typing import List

def list_of_file_names(settings_dirs: List[str], spec_option: str) -> List[str]:
    all_file_names = []
    for directory in settings_dirs:
        for root, dirs, files in os.walk(directory):
            if spec_option == 'all':
                all_file_names.extend(files)
            elif spec_option == 'python':
                all_file_names.extend([f for f in files if f.endswith('.py')])
            elif spec_option == 'txt':
                all_file_names.extend([f for f in files if f.endswith('.txt')])
            # Add more specs as needed
    return all_file_names