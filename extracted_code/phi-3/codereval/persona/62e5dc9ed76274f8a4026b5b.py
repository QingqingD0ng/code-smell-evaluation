import functools

def deprecated(message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Warning: {message}. Function {func.__name__} is deprecated.")
            return func(*args, **kwargs)
        return wrapper
    return decorator