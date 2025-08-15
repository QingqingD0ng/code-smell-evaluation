import numpy as np

def force_string(obj):
    if isinstance(obj, (np.bytes_, bytes)):
        return obj.decode('utf-8')
    return obj