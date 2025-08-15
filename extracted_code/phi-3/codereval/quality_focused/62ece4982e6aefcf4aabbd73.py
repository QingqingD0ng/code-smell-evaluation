import os
import platform

def split(s, platform='this'):
    if platform == 'this':
        platform = 0 if os.name == 'nt' else 1
    
    if platform == 0:
        return s.split(';')
    elif platform == 1:
        return s.split()
    else:
        raise ValueError("Invalid platform value. Use 0 for Windows/CMD or 1 for POSIX.")