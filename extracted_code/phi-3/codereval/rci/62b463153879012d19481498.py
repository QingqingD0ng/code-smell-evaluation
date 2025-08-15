import os

def files_list(path, file_extension=None):
    if not os.path.isdir(path):
        print("Error: Provided path is not a directory or is inaccessible.")
        return []

    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        if file_extension:
            files = [f for f in files if f.endswith(file_extension)]

        return files
    except FileNotFoundError:
        print("Error: Directory not found.")
        return []
    except PermissionError:
        print("Error: Permission denied.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []