def cachedmethod(cache, key=hash_key, lock=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            if cache_key in cache:
                return cache[cache_key]
            else:
                with lock or contextlib.nullcontext():
                    result = func(*args, **kwargs)
                    cache[cache_key] = result
                    return result
        return wrapper
    return decorator