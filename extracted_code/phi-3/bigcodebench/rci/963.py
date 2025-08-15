import os
import zipfile
from pathlib import Path

def task_func(source_directory, target_directory, zip_name):
    os.makedirs(target_directory, exist_ok=True)

    extensions = ('.txt', '.docx', '.xlsx', '.csv')

    zip_path = os.path.join(target_directory, f"{zip_name}.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(source_directory):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, Path(file_path).name)

    return zip_path