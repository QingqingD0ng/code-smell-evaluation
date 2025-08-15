import os
import sys

def is_gitbash():
    if not sys.platform.startswith('win'):
        return False

    git_for_windows_path = 'C:\\Program Files\\Git\\cmd'
    if git_for_windows_path not in os.environ['PATH']:
        return False

    if not os.path.isfile(os.path.join(git_for_windows_path, 'bash.exe')):
        return False

    return True