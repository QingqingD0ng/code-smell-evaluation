def cached(cache, key=hashkey, lock=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            k = key(*args, **kwargs)
            if lock:
                with lock:
                    if k in cache:
                        return cache[k]
            else:
                if k in cache:
                    return cache[k]

            result = func(*args, **kwargs)
            cache[k] = result
            return result
        return wrapper
    return decorator