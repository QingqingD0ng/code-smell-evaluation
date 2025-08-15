from functools import lru_cache


def memoize_lru(maxsize=128, typed=False):
    return lru_cache(maxsize=maxsize, typed=typed)