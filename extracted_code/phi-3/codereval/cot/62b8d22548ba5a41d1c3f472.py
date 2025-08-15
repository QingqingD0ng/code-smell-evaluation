import functools
import threading

def cachedmethod(cache, key=lambda *args, **kwargs: hash(frozenset(zip(args, kwargs.values()))), lock=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            if cache_key in cache:
                return cache[cache_key]
            result = func(*args, **kwargs)
            if lock:
                with lock:
                    cache[cache_key] = result
            else:
                cache[cache_key] = result
            return result
        return wrapper
    return decorator

# Example usage:
# cache = {}
# lock = threading.Lock()
# @cachedmethod(cache, lock=lock)
# def my_function(arg1, arg2):
#     # Function implementation
#     pass