from functools import lru_cache
from time import monotonic, sleep

def ttl_cache(maxsize=128, ttl=600, timer=monotonic, typed=False):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)
        timestamps = {}

        def wrapper(*args, **kwargs):
            current_time = timer()
            if args in timestamps and current_time - timestamps[args] > ttl:
                cached_func.cache_clear()
                timestamps.clear()

            result = cached_func(*args, **kwargs)
            timestamps[args] = current_time
            return result

        return wrapper
    return decorator