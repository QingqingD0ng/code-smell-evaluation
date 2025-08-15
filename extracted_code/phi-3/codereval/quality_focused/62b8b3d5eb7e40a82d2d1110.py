import ctypes

def _c_optimizations_available():
    try:
        ctypes.CDLL('libc.so.6')
        return True
    except OSError:
        return False