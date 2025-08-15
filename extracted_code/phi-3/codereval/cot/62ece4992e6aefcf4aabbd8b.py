import numpy as np

def force_string(obj):
    if isinstance(obj, (bytes, np.bytes_)):
        return obj.decode('utf-8')
    return obj