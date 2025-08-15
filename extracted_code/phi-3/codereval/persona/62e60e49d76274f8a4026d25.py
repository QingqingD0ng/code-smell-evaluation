from functools import wraps
import time

def unit_of_work(metadata=None, timeout=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time > timeout:
                raise TimeoutError(f"Function exceeded timeout of {timeout} seconds")
            if metadata:
                setattr(wrapper,'metadata', metadata)
            return result
        return wrapper
    return decorator