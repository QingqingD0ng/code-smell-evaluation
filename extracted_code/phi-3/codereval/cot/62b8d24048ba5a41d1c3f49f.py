import time
from collections import OrderedDict
from functools import wraps

def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()
        def wrapper(*args, **kwargs):
            nonlocal cache
            key = (args, tuple(sorted(kwargs.items()))) if typed else (args, tuple(sorted(kwargs.items())))
            current_time = timer()
            if key in cache and current_time - cache[key][1] < ttl:
                cache.move_to_end(key)
                return cache[key][0]
            result = user_function(*args, **kwargs)
            cache[key] = (result, current_time)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result
        return wrapper
    return decorating_function