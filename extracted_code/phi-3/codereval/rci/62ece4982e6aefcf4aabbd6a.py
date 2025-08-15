def replace_dots(value, arg):
    if not isinstance(value, str) or not isinstance(arg, str):
        raise TypeError("Both arguments must be strings")
    return value.replace('.', arg)