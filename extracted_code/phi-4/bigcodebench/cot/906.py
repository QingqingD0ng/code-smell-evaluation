import os
import zipfile

def task_func(source_dir: str, target_dir: str, archive_name: str = 'archive.zip') -> str:
    processed_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('_processed'):
                processed_files.append(os.path.join(root, file))
    
    if not processed_files:
        return os.path.join(target_dir, archive_name)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    archive_path = os.path.join(target_dir, archive_name)
    
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for file in processed_files:
            archive.write(file, os.path.relpath(file, source_dir))
    
    return archive_path