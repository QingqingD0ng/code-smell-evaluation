def cachedmethod(cache, key=lambda args: hash(args), lock=None):

    def decorator(func):

        def wrapper(*args, **kwargs):

            cache_key = key(args, kwargs)

            if lock:

                lock.acquire()

            result = cache.get(cache_key)

            if result is not None:

                if lock:

                    lock.release()

                return result

            result = func(*args, **kwargs)

            cache[cache_key] = result

            if lock:

                lock.release()

            return result

        return wrapper

    return decorator