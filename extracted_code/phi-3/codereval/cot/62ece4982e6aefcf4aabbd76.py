import os

def match(filename):
    return os.path.basename(filename).lower() == 'doxyfile'