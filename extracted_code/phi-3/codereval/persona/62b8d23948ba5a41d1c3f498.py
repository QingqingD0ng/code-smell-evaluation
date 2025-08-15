import functools

def lru_cache(maxsize=128, typed=False):
    def decorating_function(user_function):
        cache = functools.lru_cache(maxsize=maxsize, typed=typed)(user_function)
        def wrapper(*args, **kwargs):
            return cache(*args, **kwargs)
        return wrapper
    return decorating_function