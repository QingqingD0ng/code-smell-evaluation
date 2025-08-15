from collections import OrderedDict

def mru_cache(maxsize=128, typed=False):
    def decorator(user_function):
        cache = OrderedDict()

        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            # Typed support is not required, but kept for future use
            if typed:
                key = (key,)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = user_function(*args, **kwargs)
            cache[key] = result
            # Ensure cache does not exceed maxsize
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        return wrapper

    return decorator