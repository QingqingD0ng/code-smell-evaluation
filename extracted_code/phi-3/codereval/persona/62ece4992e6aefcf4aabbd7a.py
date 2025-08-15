import os

def is_gitbash():
    return os.name == 'nt' and 'GIT_BASH' in os.environ