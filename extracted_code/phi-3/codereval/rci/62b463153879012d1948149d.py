import os

def _explore_folder(folder):
    grouped_files = {}
    try:
        for filename in os.listdir(folder):
            if filename.lower().endswith('.xml'):
                file_path = os.path.join(folder, filename)
                grouped_files.setdefault(filename, []).append(file_path)
    except FileNotFoundError:
        print(f"The directory {folder} does not exist.")
    except PermissionError:
        print(f"Permission denied to access the directory {folder}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return grouped_files