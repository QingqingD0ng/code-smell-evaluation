@deprecated("This function is deprecated and will be removed in the next major release.")
def old_function():
    pass

def deprecated(message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Warning: {message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator