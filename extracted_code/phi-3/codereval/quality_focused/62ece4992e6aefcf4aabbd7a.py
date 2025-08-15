import os

def is_gitbash():
    return os.name == 'nt' and os.environ.get('TERM_PROGRAM') == 'git'