import os
import platform

def split(s, platform='this'):
    if platform == 'this':
        plat = platform.system().lower()
    elif platform == 1:
        plat = 'posix'
    elif platform == 0:
        plat = 'windows'
    else:
        raise ValueError("Invalid platform")

    if plat == 'posix':
        return s.split()
    elif plat == 'windows':
        return s.split(';')
    else:
        raise NotImplementedError("Platform not supported")