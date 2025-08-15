import os

def files_list(path):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames]