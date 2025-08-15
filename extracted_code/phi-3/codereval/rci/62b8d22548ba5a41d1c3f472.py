from functools import wraps
import time
import threading

def cachedmethod(cache, key=hashkey, lock=None, expire=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            with cache_lock:
                if cache_key in cache and (expire is None or time.time() - cache[cache_key][1] < expire):
                    return cache[cache_key][0]
            result = func(*args, **kwargs)
            with cache_lock:
                cache[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

cache = {}
cache_lock = threading.Lock()

@cachedmethod(cache, expire=10)
def my_function(a, b):
    # function implementation
    pass