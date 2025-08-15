import time
from collections import OrderedDict

def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()

        def wrapper(*args, **kwargs):
            nonlocal cache
            key = (args, tuple(sorted(kwargs.items()))) if typed else args
            current_time = timer()
            if key in cache and (current_time - cache[key][1] < ttl):
                return cache[key][0]
            result = user_function(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.popitem(last=False)
            cache[key] = (result, current_time)
            return result
        return wrapper
    return decorating_function

# Usage example:
@ttl_cache(maxsize=128, ttl=600)
def my_function(arg1, arg2):
    # Function implementation
    pass