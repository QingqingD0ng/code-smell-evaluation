def is_none_string(val: any) -> bool:

    return isinstance(val, str) and val.lower() == 'none'