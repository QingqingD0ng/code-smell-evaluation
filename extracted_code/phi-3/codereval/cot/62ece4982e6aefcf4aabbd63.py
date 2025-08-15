def is_none_string(val: any) -> bool:
    if isinstance(val, str):
        return val.lower() == 'none'
    return False