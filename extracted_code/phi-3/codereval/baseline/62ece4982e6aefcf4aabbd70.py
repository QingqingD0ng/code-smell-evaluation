import os

def os_is_mac():
    return os.uname().sysname == 'Darwin'