def cached(cache, key=hashkey, lock=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if lock:
                lock.acquire()
            key_value = key(*args, **kwargs)
            if lock:
                lock.release()
            if key_value in cache:
                return cache[key_value]
            result = func(*args, **kwargs)
            cache[key_value] = result
            return result
        return wrapper
    return decorator

# Example usage:
# cache = {}
# lock = threading.Lock()
# @cached(cache, key=lambda *args, **kwargs: your_custom_key, lock=lock)
# def your_function(arg1, arg2):
#     # Function implementation
#     pass