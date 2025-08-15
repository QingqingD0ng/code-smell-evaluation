import os

def is_gitbash():
    return os.name == 'nt' and 'MSYSTEM' in os.environ