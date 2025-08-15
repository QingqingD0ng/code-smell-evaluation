import ctypes

def _c_optimizations_available():
    try:
        c_lib = ctypes.CDLL('lib_optimizations.so')
        if hasattr(c_lib, 'optimize'):
            return c_lib
        else:
            return False
    except OSError:
        return False