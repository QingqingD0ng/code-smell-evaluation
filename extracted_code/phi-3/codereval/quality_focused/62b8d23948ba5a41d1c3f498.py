import functools
from collections import OrderedDict

def lru_cache(maxsize=128, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()

        @functools.wraps(user_function)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            if typed:
                key = (args[0].__class__,) + args[1:] + key
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = user_function(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result
        return wrapper
    return decorating_function