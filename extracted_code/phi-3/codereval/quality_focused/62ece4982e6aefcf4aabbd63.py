def is_none_string(val):
    if isinstance(val, str):
        return val.lower() == 'none'
    return False