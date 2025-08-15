import zipfile
import os
import re
import shutil

def task_func(source_dir, target_dir, archive_name='archive.zip'):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"The source directory '{source_dir}' does not exist.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pattern = re.compile(r'_processed\.')
    files_to_archive = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if pattern.search(f)]

    shutil.copytree(source_dir, os.path.join(target_dir, os.path.basename(source_dir)), ignore=shutil.ignore_patterns('*_processed*'))
    zip_file_path = os.path.join(target_dir, archive_name)

    with zipfile.ZipFile(zip_file_path, 'w') as archive:
        for file in files_to_archive:
            archive.write(file, os.path.basename(file))

    return zip_file_path

# Example usage:
# task_func('./data/', './data_processed/')
# task_func('./data/', './data_processed/','my_archive.zip')