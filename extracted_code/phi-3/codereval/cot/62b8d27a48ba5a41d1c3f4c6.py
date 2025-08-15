from functools import wraps
import hashlib

def cached(cache, key=lambda *args, **kwargs: hashlib.sha256(str(args) + str(kwargs).encode()).hexdigest(), lock=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal lock
            if lock:
                lock.acquire()
            cache_key = key(*args, **kwargs)
            if cache_key in cache:
                result = cache[cache_key]
                if lock:
                    lock.release()
                return result
            result = func(*args, **kwargs)
            cache[cache_key] = result
            if lock:
                lock.release()
            return result
        return wrapper
    return decorator