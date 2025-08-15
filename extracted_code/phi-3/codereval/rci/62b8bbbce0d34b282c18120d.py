import os

def is_file_exist(file_name):
    try:
        return os.path.isfile(file_name)
    except OSError as e:
        return False, str(e)

# Example usage
file_name = 'example.txt'
exists, error_message = is_file_exist(file_name)
if exists:
    print(f"{file_name} exists.")
else:
    print(f"Error checking if {file_name} exists: {error_message}")