import threading

def cache_decorator(cache, key_func=hashkey, lock=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = key_func(*args, **kwargs)
            if lock:
                lock.acquire()
            try:
                if cache_key in cache:
                    return cache[cache_key]
            finally:
                if lock:
                    lock.release()
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result
        return wrapper
    return decorator

# Example usage:
# cache = {}
# lock = threading.Lock()
# @cache_decorator(cache, key=lambda *args, **kwargs: str(args) + str(kwargs), lock=lock)
# def your_function(arg1, arg2):
#     # Function implementation
#     pass