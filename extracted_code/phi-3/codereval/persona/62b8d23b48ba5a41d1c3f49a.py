def mru_cache(maxsize=128, typed=False):

    cache = OrderedDict()


    def decorator(func):

        def wrapper(*args, **kwargs):

            key = args if not typed else (args, tuple(sorted(kwargs.items())))

            if key in cache:

                cache.move_to_end(key)

                return cache[key]

            result = func(*args, **kwargs)

            cache[key] = result

            if len(cache) > maxsize:

                cache.popitem(last=False)

            return result

        return wrapper

    return decorator