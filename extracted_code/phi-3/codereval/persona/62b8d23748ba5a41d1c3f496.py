import collections
import functools

def lfu_cache(maxsize=128, typed=False):
    def decorating_function(user_function):
        cache = collections.OrderedDict()
        hits = misses = 0
        if typed:
            key_func = functools.cmp_to_key(lambda a, b: (a[1] > b[1]) - (a[1] < b[1]) or (a[0] > b[0] and a[1] == b[1]) or (a[0] == b[0] and a[1] > b[1]))
        else:
            key_func = functools.cmp_to_key(lambda a, b: (a[1] > b[1]) - (a[1] < b[1]) or (a[0] > b[0]) or (a[0] == b[0]))

        @functools.wraps(user_function)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            key = args + (kwargs,) if typed else args
            if key in cache:
                hits += 1
                cache.move_to_end(key)
                return cache[key]
            misses += 1
            result = user_function(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_clear():
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0

        wrapper.cache_info = lambda: collections.OrderedDict(sorted(cache.items(), key=lambda item: item[1]))
        wrapper.hits = lambda: hits
        wrapper.misses = lambda: misses
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorating_function