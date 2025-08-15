from collections import OrderedDict

def mru_cache(maxsize=128, typed=False):
    def decorating_function(user_function):
        cache = OrderedDict()

        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            if typed:
                key = (key,)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = user_function(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.popitem(last=False)
            cache[key] = result
            return result

        return wrapper
    return decorating_function