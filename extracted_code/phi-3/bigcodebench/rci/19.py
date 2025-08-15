import os
import glob
import zipfile

def create_zip_from_directory(base_directory):
    """
    Creates a zip archive of all files in the given directory, excluding subdirectories.
    
    Parameters:
    base_directory (str): The path to the directory containing files to be zipped.
    
    Returns:
    str: The path to the created zip archive. Returns None if the directory is empty or does not exist.
    
    Raises:
    FileNotFoundError: If the base_directory does not exist.
    """
    
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"The base directory {base_directory} does not exist.")
    
    zip_archive_name = 'files.zip'
    all_files_path = os.path.join(base_directory, '*')
    all_files = glob.glob(all_files_path)
    
    if not all_files:
        return None
    
    with zipfile.ZipFile(zip_archive_name, 'w') as zipf:
        for file_path in all_files:
            if os.path.isfile(file_path):
                zipf.write(file_path, os.path.relpath(file_path, base_directory))
    
    return zip_archive_name

# Example usage:
# try:
#     zip_path = create_zip_from_directory('/path/to/files')
#     print(isinstance(zip_path, str))
# except FileNotFoundError as e:
#     print(e)