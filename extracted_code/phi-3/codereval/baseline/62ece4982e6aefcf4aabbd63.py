def is_none_string(val):
    if isinstance(val, str) and val.lower() == 'none':
        return True
    return False