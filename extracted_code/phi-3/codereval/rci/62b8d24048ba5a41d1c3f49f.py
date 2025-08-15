import time
from collections import OrderedDict

def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()

        def new_user_function(*args, **kwargs):
            key = args if not typed else (args,)
            current_time = timer()
            if key in cache and current_time - cache[key][1] < ttl:
                return cache[key][0]
            result = user_function(*args, **kwargs)
            cache[key] = (result, current_time)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result
        return new_user_function
    return decorating_function