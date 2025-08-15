def deprecated(message):
    def decorator(func):
        def new_func(*args, **kwargs):
            print(f"Warning: {func.__name__} is deprecated: {message}")
            return func(*args, **kwargs)
        return new_func
    return decorator

# Example usage:
@deprecated("Use the updated function instead.")
def old_function():
    print("This is the old function.")