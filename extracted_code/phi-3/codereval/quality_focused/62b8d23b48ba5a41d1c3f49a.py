from collections import OrderedDict

def mru_cache(maxsize=128, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()

        def wrapper(*args, **kwargs):
            key = args if not typed else (args, tuple(sorted(kwargs.items())))
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