import warnings
from functools import wraps

def deprecated(message):
    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(f"{func.__name__} is deprecated: {message}", category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return new_func
    return decorator