import os

def is_file_exist(file_name):
    return os.path.isfile(file_name)

# Example usage:
# print(is_file_exist('example.txt'))  # Replace with your file name