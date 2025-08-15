import time
from collections import OrderedDict

def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()
        cache_timestamps = OrderedDict()

        def new_user_function(*args, **kwargs):
            nonlocal cache, cache_timestamps
            key = (args, tuple(sorted(kwargs.items()))) if typed else (args,)
            current_time = timer()
            if key in cache and current_time - cache_timestamps[key] < ttl:
                cache.move_to_end(key)
                return cache[key]
            result = user_function(*args, **kwargs)
            cache[key] = result
            cache_timestamps[key] = current_time
            if len(cache) > maxsize:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
                del cache_timestamps[oldest_key]
            return result
        return new_user_function
    return decorating_function