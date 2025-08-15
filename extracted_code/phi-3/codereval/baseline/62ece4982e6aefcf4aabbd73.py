import os
import platform

def split(s, platform='this'):
    if platform == 'this':
        current_platform = platform.system()
        return split(s, platform=int(current_platform == 'Windows'))
    return s.split(os.sep if platform == 1 else '\\')